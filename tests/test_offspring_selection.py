"""Chronological parent forecasts, delayed outcomes and selection isolation."""
from copy import deepcopy
import json
from random import Random

import numpy as np
import pytest

pytest.importorskip('torch')
from automated_picbreeder.selection_strategies import (
    NoveltyPredictabilitySelectionStrategy, OffspringValueSelectionStrategy,
)


class Observer:
    def __init__(self, **kwargs): self.updates = 0
    def initialize(self, seed): pass
    def describe(self): return {'observer': 'fixture'}
    def state_hash(self): return str(self.updates)
    def prepare(self, image): return image.copy()
    def predict(self, images):
        return np.array([float(im.mean()/255) * .1 + .01/(1+self.updates) for im in images]), None
    def snapshot(self): return deepcopy(self)
    def train(self, replay, *, steps, batch_size, seed):
        self.updates += steps
        return {'updates': steps, 'sampled_examples': steps*batch_size, 'seed': seed}


class Predictor:
    def __init__(self, **kwargs): self.updates = 0
    def initialize(self, seed): self.seed = seed
    def describe(self): return {'predictor': 'fixture'}
    def state_hash(self): return str(self.updates)
    def prepare(self, images, mean, contexts):
        return [(im.copy(), c.copy()) for im, c in zip(images, contexts)]
    def predict(self, inputs): return np.linspace(0, 1, len(inputs)) / (1+self.updates)
    def train(self, replay, *, steps, batch_size, seed):
        self.updates += steps
        return {'updates': steps, 'sampled_examples': steps*batch_size, 'seed': seed}


@pytest.fixture
def controlled(monkeypatch):
    monkeypatch.setattr('automated_picbreeder.image_predictability.MaskedImageObserver', Observer)
    monkeypatch.setattr('automated_picbreeder.offspring_prediction.OffspringValuePredictor', Predictor)


def grid(): return [np.full((8, 8, 3), v, dtype=np.uint8) for v in range(0, 225, 25)]


def test_delayed_forecasts_warmup_repeated_transitions_and_copied_records(controlled):
    strategy = OffspringValueSelectionStrategy(comprehension_warmup_steps=0, warmup_targets=2, gamma=20, training_steps=1,
                                               predictor_training_steps=1)
    rng = Random(7)
    images = grid()
    decisions = []
    for t in range(5):
        decision = strategy.choose(images, rng=rng)
        meta = decision.evaluation.metadata['offspring']
        assert meta['completed_targets'] == max(t-1, 0)
        assert meta['forecast_available'] == (t > 0)
        assert meta['forecast_used'] == (t >= 3)
        if t < 2: assert meta['feedback'] is None
        else:
            previous = decisions[-1].evaluation.metadata['offspring']['pending']
            feedback = meta['feedback']
            assert feedback['origin_generation'] == t-1
            assert feedback['forecast'] == previous['forecast']
            assert feedback['predictor_state_hash'] == previous['predictor_state_hash']
            assert feedback['error'] == feedback['forecast'] - feedback['target']
            assert len(feedback['child_values']) == 8
            assert feedback['target'] == np.mean(np.array(feedback['child_values'])[:, -1])
            assert feedback['target_context']['observer_state_hash'] == str(t-1)
        if meta['forecast_used']:
            np.testing.assert_allclose(decision.scores, decision.evaluation.values[:, 2] + 20*decision.evaluation.values[:, 3])
        decisions.append(decision)
        # Repeated identical images still produce independent transition examples.
        images = [images[decision.position].copy()] * 9
    saved = json.dumps(decisions[2].evaluation.metadata, allow_nan=False)
    strategy.choose(images, rng=rng)
    assert json.dumps(decisions[2].evaluation.metadata, allow_nan=False) == saved
    assert len(strategy._transitions) == 4
    assert len(strategy._replay) == 9
    assert any(d.evaluation.metadata['offspring']['choice_changed'] for d in decisions)


@pytest.mark.parametrize('warmup', [0, 2, 10])
def test_gamma_zero_preserves_observer_choices_and_selection_rng(controlled, warmup):
    baseline = NoveltyPredictabilitySelectionStrategy(comprehension_warmup_steps=warmup, training_steps=1)
    strategy = OffspringValueSelectionStrategy(comprehension_warmup_steps=warmup, gamma=0, warmup_targets=1, training_steps=1)
    a, b = Random(7), Random(7)
    images = grid()
    for _ in range(5):
        x, y = baseline.choose(images, rng=a), strategy.choose(images, rng=b)
        assert (x.position, x.mode) == (y.position, y.mode)
        np.testing.assert_array_equal(x.scores, y.scores)
        np.testing.assert_array_equal(x.evaluation.values, y.evaluation.values[:, :2])
        assert x.evaluation.metadata['observer_state_after'] == y.evaluation.metadata['observer_state_after']
        assert a.getstate() == b.getstate()
        images = [images[x.position]] + grid()[:8]
    targets = max(5 - max(warmup, 1) - 1, 0)
    assert len(strategy._transitions) == targets
    assert strategy.predictor.updates == 10 * targets
    assert OffspringValueSelectionStrategy()._transitions == []


@pytest.mark.parametrize('warmup,targets', [(0, 1), (1, 2), (3, 2), (10, 10)])
def test_sequential_warmups_exclude_early_parents_and_train_before_forecast_selection(controlled, warmup, targets):
    strategy = OffspringValueSelectionStrategy(comprehension_warmup_steps=warmup,
        warmup_targets=targets, gamma=20, training_steps=1, predictor_training_steps=1)
    rng, images = Random(7), grid()
    first_eligible = max(warmup, 1)
    for t in range(first_eligible + targets + 2):
        decision = strategy.choose(images, rng=rng)
        meta = decision.evaluation.metadata
        offspring = meta['offspring']
        completed = max(t-first_eligible, 0)
        assert offspring['target_eligible'] == (t >= first_eligible)
        assert offspring['completed_targets'] == completed
        assert offspring['warmup_complete'] == (completed >= targets)
        assert offspring['forecast_used'] == (completed >= targets)
        assert strategy.predictor.updates == completed
        assert strategy.observer.updates == t+1
        if t < first_eligible:
            assert offspring['pending'] is None
            assert strategy._pending is None
        else:
            assert offspring['pending']['origin_generation'] == t
        if completed == 0:
            assert offspring['feedback'] is None and offspring['training'] is None
        else:
            assert offspring['feedback']['origin_generation'] == t-1
            assert offspring['feedback']['target_context']['observer_state_hash'] == str(t-1)
            assert offspring['training']['state_after'] == str(completed)
        if offspring['forecast_used']:
            np.testing.assert_allclose(decision.scores,
                np.array(offspring['current_values']) + 20*np.array(offspring['forecasts']))
        images = [images[decision.position]] + grid()[:8]
    assert all(t['origin_generation'] >= first_eligible for t in strategy._transitions)


def test_bad_chronology_rejected_before_any_training(controlled):
    strategy = OffspringValueSelectionStrategy()
    rng = Random(7)
    images = grid()
    decision = strategy.choose(images, rng=rng)
    before = strategy.observer.state_hash(), rng.getstate()
    wrong = [np.full((8,8,3), 254, np.uint8)] + images[1:]
    for bad in (images[:8], wrong, [np.zeros((4,4,3),np.uint8)]*9):
        with pytest.raises(ValueError): strategy.choose(bad, rng=rng)
        assert (strategy.observer.state_hash(), rng.getstate()) == before
    images = [images[decision.position]] + images[1:]
    strategy.choose(images, rng=rng)


@pytest.mark.parametrize('kwargs', [{'gamma': -1}, {'gamma': float('nan')}, {'gamma': True},
    {'warmup_targets': 0}, {'warmup_targets': True}, {'predictor_training_steps': 0},
    {'predictor_batch_size': 0}, {'predictor_learning_rate': float('inf')},
    {'observer_initialization': 'imagenet'}, {'comprehension_warmup_steps': -1},
    {'comprehension_warmup_steps': True}, {'comprehension_warmup_steps': 1.5}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError): OffspringValueSelectionStrategy(**kwargs)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_real_saved_replay_ancestry_and_gamma_zero_trajectory(tmp_path, device):
    import torch
    from PIL import Image
    from automated_picbreeder.experiment import ExperimentSettings, run_experiment
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        kwargs = dict(comprehension_warmup_steps=2, training_steps=1, batch_size=2, device=device)
        options = dict(**kwargs, gamma=0, warmup_targets=1, predictor_training_steps=2, predictor_batch_size=2)
        settings = ExperimentSettings(steps=4, size=8, checkpoint_every=2)
        output = tmp_path / 'offspring'
        run_experiment(selection_strategy=OffspringValueSelectionStrategy(**options), settings=settings,
                       output_dir=output, progress=None)
        baseline = tmp_path / 'baseline'
        run_experiment(selection_strategy=NoveltyPredictabilitySelectionStrategy(**kwargs), settings=settings,
                       output_dir=baseline, progress=None)
        data = json.loads((output/'session.json').read_text())
        original = json.loads((baseline/'session.json').read_text())
        records = {g['key']: g for g in data['genomes']}
        events = [e for e in data['events'] if e['action'] == 'select']
        controls = [e for e in original['events'] if e['action'] == 'select']
        replay = OffspringValueSelectionStrategy(**options)
        rng = Random(settings.resolved_selection_seed)
        def stable(value):
            if isinstance(value, dict):
                return {k: stable(v) for k,v in value.items() if not k.endswith('_seconds')}
            if isinstance(value, list): return [stable(v) for v in value]
            return value
        for t, (event, control) in enumerate(zip(events, controls)):
            assert event['displayed'] == control['displayed']
            assert event['position'] == control['position']
            assert event['evaluation']['metadata']['observer_state_after'] == control['evaluation']['metadata']['observer_state_after']
            images = [np.array(Image.open(output/records[k]['image'])) for k in event['displayed']]
            for k, image in zip(event['displayed'], images):
                np.testing.assert_array_equal(image, np.array(Image.open(baseline/records[k]['image'])))
            decision = replay.choose(images, rng=rng)
            np.testing.assert_array_equal(decision.scores, event['decision']['scores'])
            np.testing.assert_array_equal(decision.evaluation.values, event['evaluation']['values'])
            assert decision.position == event['position']
            assert stable(decision.evaluation.metadata) == stable(event['evaluation']['metadata'])
            if t:
                assert event['displayed'][0] == events[t-1]['genome']
                assert all(records[k]['parent'] == events[t-1]['genome'] for k in event['displayed'][1:])
        assert len(replay._transitions) == 1
        assert events[-1]['evaluation']['metadata']['offspring']['pending']['origin_generation'] == 3
        assert data['summary']['candidate_presentations'] == 36
        assert len(records) == 33
        for path in ('selected.png', 'source/src/automated_picbreeder/offspring_prediction.py',
                     'source/src/automated_picbreeder/offspring_value.py'):
            assert (output/path).exists()
    finally:
        torch.set_num_threads(old_threads)
