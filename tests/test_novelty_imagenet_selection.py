"""Frozen class confidence combined with previous-grid pixel novelty."""

import json
import random
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from automated_picbreeder.evaluation import Evaluation
from automated_picbreeder.experiment import ExperimentSettings, run_experiment
from automated_picbreeder.selection_strategies import (
    ImageNetSelectionStrategy, NoveltyImageNetSelectionStrategy, NoveltySelectionStrategy,
)


def solid(value):
    return np.full((8, 8, 3), value, dtype=np.uint8)


class ControlledClassifier:
    def describe(self):
        return {"evaluator": "controlled", "checkpoint_sha256": "test-checkpoint"}

    def evaluate(self, images):
        # Confidence decreases with brightness; the winning class changes.
        means = np.array([image.mean() / 255 for image in images])
        confidence = .9 - .3 * means
        values = np.column_stack((confidence, 1 - confidence))
        values[means > .5] = values[means > .5, ::-1]
        return Evaluation(values, ("class a", "class b"), self.describe())


def test_first_grid_and_weighted_ranks_then_reference_replacement():
    strategy = NoveltyImageNetSelectionStrategy(.25, evaluator=ControlledClassifier())
    rng = random.Random(7)
    state = rng.getstate()
    first = strategy.choose([solid(0), solid(0), solid(255)], rng=rng)
    assert first.position == 0 and first.mode == "greedy"
    np.testing.assert_array_equal(first.scores, [.5625, .5625, .375])
    assert first.evaluation.metadata["measurement_available"] == [False, True]
    second = strategy.choose([solid(85), solid(0), solid(255)], rng=rng)
    np.testing.assert_allclose(second.evaluation.values[:, 0], [0, 1/9, 4/9], atol=1e-15)
    np.testing.assert_array_equal(second.evaluation.metadata["novelty_ranks"], [0, .5, 1])
    np.testing.assert_array_equal(second.evaluation.metadata["confidence_ranks"], [.5, 1, 0])
    np.testing.assert_array_equal(second.scores, [.125, .625, .75])
    assert second.position == 2 and second.mode == "greedy"
    assert second.evaluation.metadata["measurement_available"] == [True, True]
    third = strategy.choose([solid(0), solid(255)], rng=rng)
    np.testing.assert_allclose(third.evaluation.values[:, 0], [(4/9)**2, (5/9)**2])
    assert third.evaluation.metadata["reference_generation"] == 1
    assert rng.getstate() == state


@pytest.mark.parametrize("weight", [0, 1])
def test_endpoint_choices_and_rng_match_existing_strategies(weight):
    evaluator = ControlledClassifier()
    strategy = NoveltyImageNetSelectionStrategy(weight, evaluator=evaluator)
    baseline = NoveltySelectionStrategy() if weight == 0 else ImageNetSelectionStrategy(evaluator=evaluator)
    rng, baseline_rng = random.Random(7), random.Random(7)
    for images in ([solid(i * 30) for i in range(9)], [solid(128)] * 9,
                   [solid(255), solid(0), solid(255)], [solid(0)]):
        actual = strategy.choose(images, rng=rng)
        expected = baseline.choose(images, rng=baseline_rng)
        assert (actual.position, actual.mode) == (expected.position, expected.mode)
        assert rng.getstate() == baseline_rng.getstate()
        if weight == 0:
            np.testing.assert_array_equal(actual.scores, expected.scores)


def test_ties_default_evaluator_reuse_and_detached_records(monkeypatch):
    evaluator = Mock(wraps=ControlledClassifier())
    factory = Mock(return_value=evaluator)
    monkeypatch.setattr("automated_picbreeder.imagenet.ImageNetEvaluator", factory)
    strategy = NoveltyImageNetSelectionStrategy()
    images = [solid(128)] * 9
    before = [image.copy() for image in images]
    first = strategy.choose(images, rng=random.Random(7))
    assert first.position == 0
    np.testing.assert_array_equal(first.scores, np.full(9, .5))
    saved = json.dumps(first.evaluation.metadata, sort_keys=True, allow_nan=False)
    strategy.choose([solid(255)] * 9, rng=random.Random(7))
    assert json.dumps(first.evaluation.metadata, sort_keys=True) == saved
    factory.assert_called_once_with()
    assert evaluator.evaluate.call_count == 2
    for image, original in zip(images, before):
        np.testing.assert_array_equal(image, original)
    description = strategy.describe()
    assert description["selection_strategy"] == "novelty-imagenet"
    assert description["comprehension_weight"] == .5
    json.dumps(description, allow_nan=False)


@pytest.mark.parametrize("weight", [-.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_weight_fails_before_model_loading(monkeypatch, weight):
    factory = Mock(side_effect=AssertionError("Must not load model"))
    monkeypatch.setattr("automated_picbreeder.imagenet.ImageNetEvaluator", factory)
    with pytest.raises(ValueError, match="comprehension_weight"):
        NoveltyImageNetSelectionStrategy(weight)
    factory.assert_not_called()


def test_invalid_grid_or_evaluation_does_not_advance_reference_or_rng():
    evaluator = Mock(wraps=ControlledClassifier())
    strategy = NoveltyImageNetSelectionStrategy(0, evaluator=evaluator)
    rng = random.Random(7)
    state = rng.getstate()
    with pytest.raises(ValueError, match="empty"):
        strategy.choose([], rng=rng)
    evaluator.evaluate.assert_not_called()
    evaluator.evaluate.return_value = Evaluation(np.zeros((8, 2)), ("a", "b"))
    with pytest.raises(ValueError, match="one row"):
        strategy.choose([solid(0)] * 9, rng=rng)
    assert rng.getstate() == state
    evaluator.evaluate.return_value = ControlledClassifier().evaluate([solid(0)] * 9)
    result = strategy.choose([solid(0)] * 9, rng=rng)
    assert result.mode == "random"
    assert result.evaluation.metadata["reference_generation"] is None


def test_saved_run_preserves_classifier_records_and_replays(tmp_path):
    output = tmp_path / "run"
    settings = ExperimentSettings(steps=3, size=8, checkpoint_every=2)
    evaluator = ControlledClassifier()
    run_experiment(selection_strategy=NoveltyImageNetSelectionStrategy(evaluator=evaluator),
                   output_dir=output, settings=settings, progress=None)
    data = json.loads((output / "session.json").read_text())
    assert data["metadata"]["selection_strategy"]["selection_strategy"] == "novelty-imagenet"
    assert data["summary"]["evaluated_images"] == 27
    assert data["summary"]["unique_candidates"] == 25
    records = {g["key"]: g for g in data["genomes"]}
    strategy = NoveltyImageNetSelectionStrategy(evaluator=evaluator)
    rng = random.Random(settings.resolved_selection_seed)
    for event in (e for e in data["events"] if e["action"] == "select"):
        images = [np.array(Image.open(output / records[key]["image"])) for key in event["displayed"]]
        result = strategy.choose(images, rng=rng)
        assert result.position == event["position"]
        assert result.mode == event["decision"]["mode"]
        np.testing.assert_array_equal(result.scores, event["decision"]["scores"])
        np.testing.assert_array_equal(result.evaluation.values, event["evaluation"]["values"])
        assert result.evaluation.metadata == event["evaluation"]["metadata"]
        classifier = event["evaluation"]["metadata"]["classifier_evaluation"]
        expected = evaluator.evaluate(images)
        assert classifier["names"] == list(expected.names)
        assert classifier["metadata"] == expected.metadata
        np.testing.assert_array_equal(classifier["values"], expected.values)
    assert len(list((output / "grids").glob("*.png"))) == 3
    assert len(list((output / "images").glob("*.png"))) == 25
