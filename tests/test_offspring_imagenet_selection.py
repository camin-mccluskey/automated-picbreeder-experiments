"""ImageNet offspring targets and shared delayed prediction without an observer."""

import json
from random import Random
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip('torch')
from automated_picbreeder.evaluation import Evaluation
from automated_picbreeder.experiment import ExperimentSettings, run_experiment
from automated_picbreeder.offspring_value import FrozenImageNetOffspringValue
from automated_picbreeder.selection_strategies import (
    NoveltyImageNetSelectionStrategy, OffspringValueImageNetSelectionStrategy,
)


def picture(value):
    return np.full((8, 8, 3), value, np.uint8)


def grid():
    return [picture(v) for v in range(0, 225, 25)]


class Classifier:
    def __init__(self):
        self.calls = []

    def describe(self):
        return {'evaluator': 'controlled', 'checkpoint_sha256': 'fixed'}

    def evaluate(self, images):
        self.calls.append(len(images))
        confidence = np.array([.5 + im.mean()/510 for im in images])
        return Evaluation(np.column_stack((confidence, 1-confidence)), ('a', 'b'), self.describe())


class Predictor:
    def __init__(self, **kwargs): self.updates = 0
    def initialize(self, seed): self.seed = seed
    def describe(self): return {'predictor': 'controlled'}
    def state_hash(self): return str(self.updates)
    def prepare(self, images, mean, contexts):
        assert contexts.shape == (9, 21)
        return [(im.copy(), c.copy()) for im, c in zip(images, contexts)]
    def predict(self, inputs): return np.linspace(0, 1, len(inputs)) / (1+self.updates)
    def train(self, replay, *, steps, batch_size, seed):
        self.updates += steps
        return {'updates': steps, 'seed': seed}


@pytest.fixture
def controlled(monkeypatch):
    monkeypatch.setattr('automated_picbreeder.offspring_prediction.OffspringValuePredictor', Predictor)
    observer = Mock(side_effect=AssertionError('ImageNet offspring must not construct a patch observer'))
    monkeypatch.setattr('automated_picbreeder.image_predictability.MaskedImageObserver', observer)
    return observer


def stable(value):
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if not k.endswith('_seconds')}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


def test_fixed_target_uses_selection_time_references_and_counts_all_children():
    mean = np.zeros((8, 8, 3))
    reference = np.linspace(0, 1, 9)
    classifier = {'checkpoint_sha256': 'fixed'}
    target = FrozenImageNetOffspringValue(reference_image=mean, novelty_reference=reference,
        confidence_reference=reference, selected_parent=picture(128), origin_generation=1,
        comprehension_weight=.5, evaluator_metadata=classifier)
    mean[:] = 1
    reference[:] = 1
    classifier['checkpoint_sha256'] = 'changed'
    # Seven low-value duplicates and one high-value child. Parent is excluded.
    result = target.evaluate_offspring([picture(128)] + [picture(0)]*7 + [picture(255)],
                                     confidence=np.array([0.]*7 + [1.]))
    assert result.metadata['mean_offspring_value'] == .125
    assert result.metadata['offspring_value'] == 1  # Max is the new-run default.
    assert result.metadata['child_count'] == 8
    assert result.metadata['evaluator']['checkpoint_sha256'] == 'fixed'
    np.testing.assert_array_equal(result.values[:, -1], [0.]*7 + [1.])
    high = target.evaluate_offspring([picture(128)] + [picture(255)]*8, confidence=np.ones(8))
    assert high.metadata['mean_offspring_value'] == 1  # Self-ranking would give .5.
    with pytest.raises(ValueError, match='parent'):
        target.evaluate_offspring([picture(127)]*9, confidence=np.ones(8))
    with pytest.raises(ValueError):
        target.evaluate_offspring([picture(128)]*9, confidence=np.full(8, np.nan))


@pytest.mark.parametrize('weight', [0, .5, 1])
@pytest.mark.parametrize('aggregation', ['max', 'mean'])
@pytest.mark.parametrize('reference', ['previous-grid-mean', 'previous-parent'])
def test_gamma_zero_matches_baseline_choices_scores_and_rng(controlled, weight, aggregation, reference):
    classifier = Classifier()
    strategy = OffspringValueImageNetSelectionStrategy(weight, evaluator=classifier, gamma=0, warmup_targets=1,
        offspring_aggregation=aggregation, novelty_reference=reference)
    baseline = NoveltyImageNetSelectionStrategy(weight, evaluator=Classifier(), novelty_reference=reference)
    a, b = Random(7), Random(7)
    images = grid()
    for t in range(5):
        expected = baseline.choose(images, rng=a)
        actual = strategy.choose(images, rng=b)
        assert (actual.position, actual.mode) == (expected.position, expected.mode)
        np.testing.assert_array_equal(actual.scores, expected.scores)
        np.testing.assert_array_equal(actual.evaluation.values[:, :2], expected.evaluation.values)
        assert a.getstate() == b.getstate()
        if t == 0:
            assert actual.evaluation.metadata['measurement_available'] == [False, True, True, False]
        images = [images[actual.position]] + grid()[:8]
    assert len(strategy._transitions) == 3
    assert classifier.calls == [9]*5
    controlled.assert_not_called()


@pytest.mark.parametrize('warmup', [1, 2, 10])
def test_warmup_delayed_targets_and_original_forecasts(controlled, warmup):
    strategy = OffspringValueImageNetSelectionStrategy(evaluator=Classifier(), warmup_targets=warmup,
        gamma=20, predictor_training_steps=1)
    rng, images, decisions = Random(7), grid(), []
    for t in range(warmup + 3):
        decision = strategy.choose(images, rng=rng)
        meta = decision.evaluation.metadata['offspring']
        assert meta['completed_targets'] == max(t-1, 0)
        assert meta['target_eligible'] == (t >= 1)
        assert meta['forecast_used'] == (t >= warmup+1)
        assert strategy.predictor.updates == max(t-1, 0)
        if t < 2:
            assert meta['feedback'] is None
        else:
            prior = decisions[-1].evaluation.metadata['offspring']['pending']
            feedback = meta['feedback']
            assert feedback['forecast'] == prior['forecast']
            assert feedback['predictor_state_hash'] == prior['predictor_state_hash']
            assert feedback['error'] == prior['forecast'] - feedback['target']
            assert feedback['target_context']['reference_image_hash'] == prior['reference_image_hash']
            assert len(feedback['child_values']) == 8
        if meta['forecast_used']:
            np.testing.assert_allclose(decision.scores,
                decision.evaluation.values[:, 2] + 20*decision.evaluation.values[:, 3])
        decisions.append(decision)
        images = [images[decision.position]]*9
    assert any(d.evaluation.metadata['offspring']['choice_changed'] for d in decisions)
    assert len(strategy._transitions) == warmup+1  # Repeated parents remain separate transitions.
    assert decisions[-1].evaluation.metadata['offspring']['pending']['origin_generation'] == warmup+2


def test_invalid_chronology_fails_before_classification_or_training(controlled):
    classifier = Classifier()
    strategy = OffspringValueImageNetSelectionStrategy(evaluator=classifier)
    rng, images = Random(7), grid()
    strategy.choose(images, rng=rng)
    state = rng.getstate()
    for invalid in (images[:8], [picture(254)]*9, [np.zeros((4, 4, 3), np.uint8)]*9):
        with pytest.raises(ValueError):
            strategy.choose(invalid, rng=rng)
    assert classifier.calls == [9]
    assert strategy.predictor.updates == 0
    assert rng.getstate() == state


@pytest.mark.parametrize('options', [{'gamma': -1}, {'gamma': float('nan')}, {'gamma': True},
    {'warmup_targets': 0}, {'predictor_batch_size': 0}, {'predictor_training_steps': 0},
    {'offspring_aggregation': 'median'}, {'offspring_aggregation': None},
    {'predictor_learning_rate': float('inf')}, {'comprehension_weight': -1}, {'device': 'cuda'}])
def test_invalid_configuration_before_loading_classifier(monkeypatch, options):
    factory = Mock(side_effect=AssertionError('Do not load classifier'))
    monkeypatch.setattr('automated_picbreeder.imagenet.ImageNetEvaluator', factory)
    with pytest.raises(ValueError):
        OffspringValueImageNetSelectionStrategy(**options)
    factory.assert_not_called()


@pytest.mark.parametrize('gamma', [0, 3])
@pytest.mark.parametrize('aggregation,reference', [('mean', 'previous-grid-mean'), ('max', 'previous-parent')])
def test_saved_run_real_predictor_replays_and_preserves_ancestry(tmp_path, gamma, aggregation, reference):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        settings = ExperimentSettings(steps=5, size=8, checkpoint_every=2)
        options = dict(gamma=gamma, warmup_targets=1, predictor_training_steps=2, predictor_batch_size=2,
                       offspring_aggregation=aggregation, novelty_reference=reference)
        run_experiment(selection_strategy=OffspringValueImageNetSelectionStrategy(evaluator=Classifier(), **options),
                       settings=settings, output_dir=tmp_path/'run', progress=None)
        data = json.loads((tmp_path/'run/session.json').read_text())
        events = [e for e in data['events'] if e['action'] == 'select']
        records = {g['key']: g for g in data['genomes']}
        replay = OffspringValueImageNetSelectionStrategy(evaluator=Classifier(), **options)
        rng = Random(settings.resolved_selection_seed)
        for t, event in enumerate(events):
            images = [np.array(Image.open(tmp_path/'run'/records[k]['image'])) for k in event['displayed']]
            decision = replay.choose(images, rng=rng)
            assert decision.position == event['position']
            np.testing.assert_array_equal(decision.scores, event['decision']['scores'])
            np.testing.assert_array_equal(decision.evaluation.values, event['evaluation']['values'])
            assert stable(decision.evaluation.metadata) == stable(event['evaluation']['metadata'])
            if t:
                assert event['displayed'][0] == events[t-1]['genome']
                assert all(records[k]['parent'] == events[t-1]['genome'] for k in event['displayed'][1:])
        assert len(replay._transitions) == 3
        assert data['summary']['evaluated_images'] == 45
        assert len(records) == 41
        if gamma == 0:
            run_experiment(selection_strategy=NoveltyImageNetSelectionStrategy(evaluator=Classifier(), novelty_reference=reference),
                           settings=settings, output_dir=tmp_path/'control', progress=None)
            control = json.loads((tmp_path/'control/session.json').read_text())
            assert data['genomes'] == control['genomes']
            assert [e['genome'] for e in events] == [e['genome'] for e in control['events'] if e['action'] == 'select']
    finally:
        torch.set_num_threads(old_threads)
