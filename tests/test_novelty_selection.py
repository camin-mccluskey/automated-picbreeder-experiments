"""Pixel novelty uses the previous displayed grid, independent of the observer."""

import json
import random

import numpy as np
import pytest
from PIL import Image

from automated_picbreeder.experiment import ExperimentSettings, run_experiment
from automated_picbreeder.selection_strategies import NoveltySelectionStrategy


def solid(value, shape=(2, 2, 3)):
    return np.full(shape, value, dtype=np.uint8)


def test_first_grid_is_uniform_and_neutral_with_no_global_rng_changes():
    strategy = NoveltySelectionStrategy()
    rng, expected = random.Random(7), random.Random(7)
    global_state = random.getstate()
    numpy_state = np.random.get_state()
    decision = strategy.choose([solid(i) for i in range(9)], rng=rng)
    assert decision.position == expected.randrange(9)
    assert decision.mode == "random"
    assert rng.getstate() == expected.getstate()
    assert random.getstate() == global_state
    for a, b in zip(numpy_state, np.random.get_state()):
        np.testing.assert_equal(a, b)
    np.testing.assert_array_equal(decision.evaluation.values, np.zeros((9, 1)))
    np.testing.assert_array_equal(decision.scores, np.full(9, .5))
    assert decision.evaluation.metadata["reference_generation"] is None
    assert not decision.evaluation.metadata["reference_available"]
    decision.validate(9)
    json.dumps(strategy.describe(), allow_nan=False)


def test_mean_includes_duplicates_and_replaces_older_generation():
    strategy, rng = NoveltySelectionStrategy(), random.Random(7)
    strategy.choose([solid(0), solid(0), solid(255)], rng=rng)
    before = rng.getstate()
    decision = strategy.choose([solid(85), solid(0), solid(255)], rng=rng)
    np.testing.assert_allclose(decision.evaluation.values[:, 0], [0, 1/9, 4/9], atol=1e-15)
    np.testing.assert_array_equal(decision.scores, [0, .5, 1])
    assert decision.position == 2
    assert decision.mode == "greedy"
    assert rng.getstate() == before
    assert decision.evaluation.metadata["seen_before"] == [False, True, True]
    assert decision.evaluation.metadata["reference_generation"] == 0
    # The new reference is 340/3, not the initial mean or a running mean.
    next_decision = strategy.choose([solid(0), solid(255)], rng=rng)
    np.testing.assert_allclose(next_decision.evaluation.values[:, 0], [(4/9)**2, (5/9)**2])
    assert next_decision.evaluation.metadata["reference_generation"] == 1


def test_mean_preserves_pixel_positions_and_channels_without_uint8_overflow():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, :, 0] = 255
    strategy, rng = NoveltySelectionStrategy(), random.Random(1)
    strategy.choose([image], rng=rng)
    shifted = np.roll(image, 1, axis=0)
    decision = strategy.choose([image, shifted, 255 - image], rng=rng)
    np.testing.assert_allclose(decision.evaluation.values[:, 0], [0, 1/3, 1])
    assert decision.position == 2


def test_average_ranks_and_first_maximum_tie():
    strategy, rng = NoveltySelectionStrategy(), random.Random(1)
    strategy.choose([solid(0)], rng=rng)
    decision = strategy.choose([solid(0), solid(128), solid(128), solid(255), solid(255)], rng=rng)
    np.testing.assert_allclose(decision.scores, [0, .375, .375, .875, .875])
    assert decision.position == 3
    tied = strategy.choose([solid(0)] * 9, rng=rng)
    np.testing.assert_array_equal(tied.scores, np.full(9, .5))
    assert tied.position == 0
    singleton = strategy.choose([solid(255)], rng=rng)
    np.testing.assert_array_equal(singleton.scores, [.5])


def test_inputs_configuration_and_earlier_records_are_detached():
    strategy, rng = NoveltySelectionStrategy(), random.Random(1)
    config = strategy.describe()
    images = [solid(0), solid(255)]
    before = [image.copy() for image in images]
    first = strategy.choose(images, rng=rng)
    saved = json.dumps(first.evaluation.metadata, sort_keys=True)
    for a, b in zip(images, before):
        np.testing.assert_array_equal(a, b)
    images[0][:] = 255  # Mutating the caller's input cannot change the reference.
    second = strategy.choose([solid(0), solid(255)], rng=rng)
    np.testing.assert_allclose(second.evaluation.values[:, 0], [.25, .25])
    assert json.dumps(first.evaluation.metadata, sort_keys=True) == saved
    assert strategy.describe() == config
    assert NoveltySelectionStrategy().choose([solid(0)], rng=rng).mode == "random"


@pytest.mark.parametrize("invalid", [[], [solid(0), solid(0, (3, 3, 3))],
                                     [np.zeros((2, 2, 3), dtype=float)]])
def test_invalid_grid_does_not_advance_state_or_rng(invalid):
    strategy, rng = NoveltySelectionStrategy(), random.Random(7)
    before = rng.getstate()
    with pytest.raises(ValueError):
        strategy.choose(invalid, rng=rng)
    assert rng.getstate() == before
    assert strategy.choose([solid(0)], rng=rng).mode == "random"
    with pytest.raises(ValueError, match="shape"):
        strategy.choose([solid(0, (3, 3, 3))], rng=rng)
    decision = strategy.choose([solid(255)], rng=rng)
    assert decision.evaluation.values[0, 0] == 1
    assert decision.evaluation.metadata["reference_generation"] == 0


def test_grayscale_contract_is_supported_without_changing_shape():
    strategy, rng = NoveltySelectionStrategy(), random.Random(1)
    strategy.choose([solid(0, (2, 2))], rng=rng)
    assert strategy.choose([solid(255, (2, 2))], rng=rng).evaluation.values[0, 0] == 1


@pytest.mark.parametrize("reference", ["previous-grid-mean", "previous-parent"])
def test_saved_run_replays_scores_choices_and_images(tmp_path, reference):
    settings = ExperimentSettings(steps=3, size=8, checkpoint_every=2)
    run_experiment(selection_strategy=NoveltySelectionStrategy(novelty_reference=reference), output_dir=tmp_path / "run",
                   settings=settings, progress=None)
    data = json.loads((tmp_path / "run/session.json").read_text())
    assert data["metadata"]["selection_strategy"]["selection_strategy"] == "novelty"
    assert data["summary"]["candidate_presentations"] == 27
    assert data["summary"]["evaluated_images"] == 27
    assert data["summary"]["unique_candidates"] == 25
    records = {g["key"]: g for g in data["genomes"]}
    strategy, rng = NoveltySelectionStrategy(novelty_reference=reference), random.Random(settings.resolved_selection_seed)
    decisions = [e for e in data["events"] if e["action"] == "select"]
    for event in decisions:
        images = [np.array(Image.open(tmp_path / "run" / records[key]["image"])) for key in event["displayed"]]
        result = strategy.choose(images, rng=rng)
        assert result.position == event["position"]
        assert result.mode == event["decision"]["mode"]
        np.testing.assert_array_equal(result.scores, event["decision"]["scores"])
        np.testing.assert_array_equal(result.evaluation.values, event["evaluation"]["values"])
        assert result.evaluation.metadata == event["evaluation"]["metadata"]
    assert len(list((tmp_path / "run/images").glob("*.png"))) == 25
    assert len(list((tmp_path / "run/grids").glob("*.png"))) == 3
    assert len(list((tmp_path / "run/checkpoints").glob("*.json"))) == 2
    assert (tmp_path / "run/selected.png").read_bytes() == (tmp_path / "run" / records[data["selected"]]["image"]).read_bytes()
