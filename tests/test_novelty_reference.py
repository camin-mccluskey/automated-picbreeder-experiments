"""Configured local novelty uses the preceding selected parent or grid mean."""

from random import Random

import numpy as np
import pytest

from automated_picbreeder.selection_strategies import (
    NoveltySelectionStrategy, NoveltyImageNetSelectionStrategy,
    NoveltyPredictabilitySelectionStrategy, OffspringValueSelectionStrategy,
    OffspringValueImageNetSelectionStrategy,
)


def solid(value):
    return np.full((4, 4, 3), value, dtype=np.uint8)


@pytest.mark.parametrize("reference", ["previous-grid-mean", "previous-parent"])
def test_first_choice_and_rng_are_unchanged(reference):
    rng, expected = Random(7), Random(7)
    strategy = NoveltySelectionStrategy(novelty_reference=reference)
    decision = strategy.choose([solid(v) for v in range(9)], rng=rng)
    assert decision.position == expected.randrange(9)
    assert rng.getstate() == expected.getstate()
    assert decision.mode == "random"
    np.testing.assert_array_equal(decision.scores, np.full(9, .5))
    np.testing.assert_array_equal(decision.evaluation.values, np.zeros((9, 1)))
    assert decision.evaluation.metadata["novelty_reference"] == reference
    assert strategy.describe()["novelty_reference"] == reference


def test_previous_parent_changes_choice_and_has_zero_self_distance():
    # Seed 5 chooses white (position 2), while duplicate black images dominate the mean.
    grid = [solid(0), solid(0), solid(255)]
    mean = NoveltySelectionStrategy()
    parent = NoveltySelectionStrategy(novelty_reference="previous-parent")
    for strategy in (mean, parent):
        assert strategy.choose(grid, rng=Random(5)).position == 2
    next_grid = [solid(255), solid(0), solid(85)]
    mean_decision = mean.choose(next_grid, rng=Random(5))
    parent_decision = parent.choose(next_grid, rng=Random(5))
    assert mean_decision.position == 0
    assert parent_decision.position == 1
    np.testing.assert_allclose(parent_decision.evaluation.values[:, 0], [0, 1, (2/3)**2])
    # The selected black child becomes the next reference, replacing white.
    next_decision = parent.choose([solid(0), solid(255)], rng=Random(5))
    np.testing.assert_array_equal(next_decision.evaluation.values[:, 0], [0, 1])


def test_parent_reference_is_detached_from_caller_and_only_uses_selected_image():
    images = [solid(0), solid(255), solid(255)]
    strategy = NoveltySelectionStrategy(novelty_reference="previous-parent")
    first = strategy.choose(images, rng=Random(1))
    assert first.position == 0
    images[0][:] = 255
    second = strategy.choose([solid(0), solid(255)], rng=Random(1))
    np.testing.assert_array_equal(second.evaluation.values[:, 0], [0, 1])


def test_explicit_default_reproduces_implicit_default():
    implicit = NoveltySelectionStrategy()
    explicit = NoveltySelectionStrategy(novelty_reference="previous-grid-mean")
    a, b = Random(7), Random(7)
    for values in ([0, 0, 255], [100, 30, 200], [255, 0, 0]):
        images = [solid(v) for v in values]
        x, y = implicit.choose(images, rng=a), explicit.choose(images, rng=b)
        assert x.position == y.position
        np.testing.assert_array_equal(x.evaluation.values, y.evaluation.values)
        assert x.evaluation.metadata == y.evaluation.metadata
        assert a.getstate() == b.getstate()


@pytest.mark.parametrize("strategy_class", [NoveltySelectionStrategy, NoveltyImageNetSelectionStrategy,
    NoveltyPredictabilitySelectionStrategy, OffspringValueSelectionStrategy,
    OffspringValueImageNetSelectionStrategy])
@pytest.mark.parametrize("invalid", ["all-history", None, True, 1])
def test_invalid_reference_fails_before_model_construction(monkeypatch, strategy_class, invalid):
    # Constructors import optional model modules only after validating the option.
    import builtins
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name in ("imagenet", "image_predictability", "offspring_prediction"):
            raise AssertionError("Invalid reference must fail before importing model helpers")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError, match="novelty_reference"):
        strategy_class(novelty_reference=invalid)
