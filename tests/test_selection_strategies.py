"""Selection strategies can be exercised without Torch or downloaded weights."""

import json
import random
import subprocess
import sys
from unittest.mock import Mock

import numpy as np
import pytest

from automated_picbreeder.evaluation import Evaluation
from automated_picbreeder.selection_strategies import (
    ImageNetSelectionStrategy,
    RandomSelectionStrategy,
    SelectionDecision,
    SelectionStrategy,
)


@pytest.fixture
def images():
    return [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(9)]


@pytest.fixture
def evaluator():
    # Valid, controlled class probabilities, including a tie across classes.
    values = np.full((9, 1000), 0.001)
    for row, column, confidence in ((0, 3, .4), (1, 5, .6), (2, 8, .6)):
        values[row] = (1 - confidence) / 999
        values[row, column] = confidence
    result = Evaluation(values, tuple(f"class {i}" for i in range(1000)),
                        {"evaluator": "controlled_imagenet"})
    return Mock(evaluate=Mock(return_value=result),
                describe=Mock(return_value={"evaluator": "controlled_imagenet"}))


def test_greedy_uses_best_class_per_image_and_first_tie(images, evaluator):
    strategy: SelectionStrategy = ImageNetSelectionStrategy(evaluator=evaluator)
    rng = random.Random(7)
    before_rng = rng.getstate()
    before_images = [image.copy() for image in images]
    before_values = evaluator.evaluate.return_value.values.copy()

    decision = strategy.choose(images, rng=rng)

    assert decision.position == 1
    assert decision.mode == "greedy"
    assert decision.evaluation.names[5] == "class 5"
    assert decision.evaluation.values.shape == (9, 1000)
    np.testing.assert_array_equal(decision.scores, [.4, .6, .6, .001, .001, .001, .001, .001, .001])
    np.testing.assert_array_equal(decision.evaluation.values, before_values)
    evaluator.evaluate.assert_called_once_with(images)
    assert rng.getstate() == before_rng
    for before, after in zip(before_images, images, strict=True):
        np.testing.assert_array_equal(before, after)

    evaluator.evaluate.return_value.values[0] = before_values[1]
    assert strategy.choose(images, rng=rng).position == 0  # Parent wins ties.


@pytest.mark.parametrize("draw,position,mode", [(.099, 4, "random"), (.1, 1, "greedy"), (.8, 1, "greedy")])
def test_epsilon_branches_share_measurements_and_scoring(images, evaluator, draw, position, mode):
    rng = Mock(random=Mock(return_value=draw), randrange=Mock(return_value=4))
    decision = ImageNetSelectionStrategy(epsilon=.1, evaluator=evaluator).choose(images, rng=rng)
    assert decision.position == position
    assert decision.mode == mode
    np.testing.assert_array_equal(decision.scores, evaluator.evaluate.return_value.values.max(axis=1))
    evaluator.evaluate.assert_called_once_with(images)  # Also evaluated on exploration.
    if mode == "random":
        rng.randrange.assert_called_once_with(9)
    else:
        rng.randrange.assert_not_called()


def test_exploration_is_recorded_even_when_it_selects_the_greedy_winner(images, evaluator):
    rng = Mock(random=Mock(return_value=.01), randrange=Mock(return_value=1))
    decision = ImageNetSelectionStrategy(epsilon=.1, evaluator=evaluator).choose(images, rng=rng)
    assert decision.position == 1
    assert decision.mode == "random"


@pytest.mark.parametrize("position", [0, 8])
def test_uniform_random_includes_parent_and_last_child_without_evaluation(images, position):
    rng = Mock(randrange=Mock(return_value=position))
    decision = RandomSelectionStrategy().choose(images, rng=rng)
    assert decision.position == position
    assert decision.mode == "random"
    assert decision.evaluation is None
    assert decision.scores is None
    rng.randrange.assert_called_once_with(9)
    rng.random.assert_not_called()


def test_epsilon_one_matches_uniform_random_and_strategy_reuse_is_reproducible(images, evaluator):
    uniform = RandomSelectionStrategy()
    scored = ImageNetSelectionStrategy(epsilon=1, evaluator=evaluator)
    global_state = random.getstate()
    trajectories = []
    for _ in range(2):
        a, b = random.Random(71), random.Random(71)
        plain = [uniform.choose(images, rng=a).position for _ in range(12)]
        measured = [scored.choose(images, rng=b).position for _ in range(12)]
        assert plain == measured
        assert a.getstate() == b.getstate()  # No redundant exploration coin draw.
        trajectories.append(plain)
    assert trajectories[0] == trajectories[1]
    assert random.getstate() == global_state


@pytest.mark.parametrize("epsilon", [-.1, 1.1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_epsilon_fails_before_loading_a_classifier(monkeypatch, epsilon):
    factory = Mock(side_effect=AssertionError("Must not load a classifier"))
    monkeypatch.setattr("automated_picbreeder.imagenet.ImageNetEvaluator", factory)
    with pytest.raises(ValueError, match="epsilon"):
        ImageNetSelectionStrategy(epsilon=epsilon)
    factory.assert_not_called()


def test_default_strategy_reuses_one_classifier_and_describes_configuration(monkeypatch, images, evaluator):
    factory = Mock(return_value=evaluator)
    monkeypatch.setattr("automated_picbreeder.imagenet.ImageNetEvaluator", factory)
    strategy = ImageNetSelectionStrategy(epsilon=.2)
    strategy.choose(images, rng=random.Random(1))
    strategy.choose(images, rng=random.Random(2))
    factory.assert_called_once_with()
    metadata = strategy.describe()
    assert metadata["selection_strategy"] == "imagenet"
    assert metadata["epsilon"] == .2
    assert metadata["evaluator"] == evaluator.describe.return_value
    json.dumps(metadata)
    assert RandomSelectionStrategy().describe()["selection_strategy"] == "random"


def test_empty_grid_and_wrong_evaluation_rows_fail_before_selection(images, evaluator):
    strategy = ImageNetSelectionStrategy(evaluator=evaluator)
    rng = Mock()
    for candidate in (RandomSelectionStrategy(), strategy):
        with pytest.raises(ValueError, match="empty"):
            candidate.choose([], rng=rng)
    evaluator.evaluate.assert_not_called()
    evaluator.evaluate.return_value = Evaluation(np.zeros((8, 1)), ("score",))
    with pytest.raises(ValueError, match="one row"):
        strategy.choose(images, rng=rng)
    rng.random.assert_not_called()
    rng.randrange.assert_not_called()


@pytest.mark.parametrize("scores", [np.zeros((9, 1)), [float("nan")], [float("inf")], [1j], ["score"]])
def test_decision_rejects_ambiguous_or_nonfinite_scores(scores):
    with pytest.raises(ValueError, match="scores"):
        SelectionDecision(position=0, mode="greedy", scores=scores)


@pytest.mark.parametrize("position", [-1, True, 0.5])
def test_decision_rejects_invalid_positions(position):
    with pytest.raises(ValueError, match="position"):
        SelectionDecision(position=position, mode="random")


def test_decision_validates_against_grid_and_detaches_scores():
    values = np.array([.4, .6])
    decision = SelectionDecision(position=1, mode="greedy", scores=values)
    values[:] = 0
    np.testing.assert_array_equal(decision.scores, [.4, .6])
    decision.validate(2)
    with pytest.raises(ValueError, match="position"):
        decision.validate(1)
    with pytest.raises(ValueError, match="score"):
        decision.validate(3)
    with pytest.raises(ValueError, match="one row"):
        SelectionDecision(position=0, mode="custom",
                          evaluation=Evaluation(np.zeros((1, 1)), ("measurement",))).validate(2)


def test_random_selection_imports_and_runs_with_torch_unavailable():
    code = """
import builtins
import random
import sys
import numpy as np
original_import = builtins.__import__
def without_torch(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision'}:
        raise ImportError('Torch deliberately unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_torch
from automated_picbreeder.selection_strategies import RandomSelectionStrategy
images = [np.zeros((2, 2, 3), dtype=np.uint8)] * 9
decision = RandomSelectionStrategy().choose(images, rng=random.Random(7))
assert 0 <= decision.position < 9
assert decision.evaluation is None
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
