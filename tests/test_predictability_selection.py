"""Selection ordering and audit records with a controlled online observer."""

import json
from random import Random

import numpy as np
import pytest

pytest.importorskip("torch")

from automated_picbreeder.selection_strategies import NoveltySelectionStrategy, NoveltyPredictabilitySelectionStrategy


class Observer:
    def __init__(self, **kwargs):
        self.updates = 0
        self.calls = []

    def initialize(self, seed):
        self.calls.append(('initialize', seed))

    def describe(self):
        return {'observer': 'controlled'}

    def state_hash(self):
        return str(self.updates)

    def prepare(self, image):
        return image.copy()

    def predict(self, images):
        self.calls.append(('predict', self.updates))
        return np.linspace(.9, .1, len(images)), np.zeros((len(images), 32, 32, 3))

    def train(self, replay, *, steps, batch_size, seed):
        self.calls.append(('train', len(replay)))
        self.updates += steps
        return {'updates': steps, 'sampled_examples': steps * batch_size, 'losses': [.1] * steps}


@pytest.fixture
def controlled(monkeypatch):
    monkeypatch.setattr('automated_picbreeder.image_predictability.MaskedImageObserver', Observer)


def grid(values):
    return [np.full((4, 4, 3), v, dtype=np.uint8) for v in values]


def test_scores_precede_training_and_rejected_images_enter_deduplicated_replay(controlled):
    strategy = NoveltyPredictabilitySelectionStrategy(comprehension_warmup_steps=0, training_steps=2, batch_size=2)
    rng = Random(7)
    first = strategy.choose(grid([0, 0, 255]), rng=rng)
    assert first.position == Random(7).randrange(3)
    assert first.mode == 'random'
    assert first.evaluation.names == ('pixel_novelty', 'comprehension')
    assert not first.evaluation.metadata['comprehension_available']
    assert first.evaluation.metadata['masked_mse'] == [None] * 3
    saved = json.dumps(first.evaluation.metadata)
    second = strategy.choose(grid([255, 128, 0]), rng=rng)
    assert strategy.observer.calls[1:] == [('train', 2), ('predict', 2), ('train', 3)]
    meta = second.evaluation.metadata
    assert meta['observer_state_before'] == '2' and meta['observer_state_after'] == '4'
    assert meta['replay_size_before'] == 2 and meta['replay_size_after'] == 3
    np.testing.assert_allclose(second.evaluation.values[:, 1], [.1, .5, .9])
    np.testing.assert_allclose(second.scores, [.5, .25, .75])
    assert second.position == 2
    assert second.evaluation.metadata['seen_before'] == [True, False, True]
    assert json.dumps(first.evaluation.metadata) == saved
    json.dumps(second.evaluation.metadata, allow_nan=False)


@pytest.mark.parametrize('warmup', [0, 1, 3, 10])
def test_comprehension_warmup_keeps_training_and_switches_scores_after_completed_decisions(controlled, warmup):
    strategy = NoveltyPredictabilitySelectionStrategy(comprehension_warmup_steps=warmup, training_steps=2)
    rng = Random(7)
    assert strategy.describe()['comprehension_warmup_steps'] == warmup
    decisions = []
    for t in range(max(warmup, 1) + 2):
        decision = strategy.choose(grid([255, 0]), rng=rng)
        meta = decision.evaluation.metadata
        weight = .5 if t > 0 and t >= warmup else 0
        assert meta['comprehension_warmup_complete'] == (t >= warmup)
        assert meta['effective_comprehension_weight'] == weight
        assert meta['comprehension_available'] == (t > 0)
        assert strategy.observer.updates == 2 * (t + 1)
        np.testing.assert_allclose(decision.scores,
            (1-weight)*np.array(meta['novelty_ranks']) + weight*np.array(meta['comprehension_ranks']))
        decisions.append(decision)
    # Equal novelty at the endpoints: comprehension changes the choice at activation.
    if warmup > 1:
        assert decisions[warmup-1].position == 0
        assert decisions[warmup].position == 1


def test_zero_weight_matches_novelty_and_fresh_instances_have_no_history(controlled):
    plain, combined = NoveltySelectionStrategy(), NoveltyPredictabilitySelectionStrategy(comprehension_weight=0)
    a, b = Random(9), Random(9)
    for images in [grid([0, 0, 255]), grid([255, 85, 0]), grid([0] * 9)]:
        x, y = plain.choose(images, rng=a), combined.choose(images, rng=b)
        assert x.position == y.position and x.mode == y.mode
        np.testing.assert_array_equal(x.scores, y.scores)
        np.testing.assert_array_equal(x.evaluation.values[:, 0], y.evaluation.values[:, 0])
    fresh = NoveltyPredictabilitySelectionStrategy()
    assert fresh.choose(grid([0]), rng=Random(7)).mode == 'random'


@pytest.mark.parametrize('kwargs', [{'comprehension_weight': -1}, {'comprehension_weight': float('nan')},
    {'comprehension_weight': True}, {'observer_initialization': 'bad'}, {'training_steps': 0},
    {'batch_size': 1}, {'learning_rate': float('inf')},
    {'comprehension_warmup_steps': -1}, {'comprehension_warmup_steps': True},
    {'comprehension_warmup_steps': 1.5}])
def test_invalid_configuration_before_observer_construction(kwargs):
    with pytest.raises(ValueError):
        NoveltyPredictabilitySelectionStrategy(**kwargs)


def test_real_observer_saved_run_replay_and_breeding_rng_isolation(tmp_path):
    import torch
    from PIL import Image
    from automated_picbreeder.experiment import ExperimentSettings, run_experiment
    from automated_picbreeder.breeding import BreedingSession
    from automated_picbreeder.cppn import render

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        settings = ExperimentSettings(steps=3, size=8, checkpoint_every=2)
        strategy = NoveltyPredictabilitySelectionStrategy(comprehension_warmup_steps=2, training_steps=1, batch_size=2)
        run_experiment(selection_strategy=strategy, settings=settings, output_dir=tmp_path / 'run', progress=None)
        data = json.loads((tmp_path / 'run/session.json').read_text())
        records = {g['key']: g for g in data['genomes']}
        replay = NoveltyPredictabilitySelectionStrategy(comprehension_warmup_steps=2, training_steps=1, batch_size=2)
        rng = Random(settings.resolved_selection_seed)
        breeding = BreedingSession(seed=settings.seed)
        decisions = [e for e in data['events'] if e['action'] == 'select']
        for step, event in enumerate(decisions):
            if step:
                breeding.evolve(strength=settings.mutation_strength, topology=settings.topology)
            images = [np.array(Image.open(tmp_path / 'run' / records[k]['image'])) for k in event['displayed']]
            assert breeding.candidates == event['displayed']
            for key, image in zip(breeding.candidates, images):
                np.testing.assert_array_equal(render(breeding.genomes[key], breeding.config, settings.size), image)
            result = replay.choose(images, rng=rng)
            assert result.position == event['position']
            np.testing.assert_array_equal(result.scores, event['decision']['scores'])
            np.testing.assert_array_equal(result.evaluation.values, event['evaluation']['values'])
            def stable(metadata):
                return {k: v for k, v in metadata.items() if k not in ('scoring_seconds', 'training_seconds')}
            assert stable(result.evaluation.metadata) == stable(event['evaluation']['metadata'])
            breeding.select(event['position'])
        assert len(records) == 25
        assert data['summary']['evaluated_images'] == 27
    finally:
        torch.set_num_threads(threads)
