"""Reporting measures saved outcomes without changing selection or running models."""

import json
from random import Random

import numpy as np
import pytest
from PIL import Image

from automated_picbreeder.experiment import ExperimentSettings, run_experiment
from automated_picbreeder.experiment_metrics import build_run_metrics, summarize_batch
from automated_picbreeder.selection_strategies import (
    ImageNetSelectionStrategy, NoveltyImageNetSelectionStrategy,
    NoveltySelectionStrategy, RandomSelectionStrategy,
)
from automated_picbreeder.evaluation import Evaluation


class Classifier:
    def describe(self):
        return {"evaluator": "controlled"}

    def evaluate(self, images):
        confidence = np.linspace(.5, .9, len(images))
        # Names deliberately repeat: class identity must use column position.
        return Evaluation(np.column_stack((confidence, 1 - confidence)), ("same", "same"))


def run(tmp_path, strategy, **kwargs):
    run_experiment(selection_strategy=strategy, output_dir=tmp_path,
                   settings=ExperimentSettings(size=4, steps=3, **kwargs), progress=None)
    return json.loads((tmp_path / "metrics.json").read_text())


def test_common_metrics_use_pixels_and_exclude_first_grid_from_transitions(tmp_path):
    report = run(tmp_path / "run", RandomSelectionStrategy(), mutation_strength=0, topology=False)
    rows = report["generations"]
    assert rows[0]["metrics"]["pixel_mse_previous"] is None
    assert rows[0]["metrics"]["parent_retained"] is None
    assert rows[0]["metrics"]["children_identical_to_parent_fraction"] is None
    for row in rows[1:]:
        assert row["metrics"]["pixel_mse_previous"] == 0
        assert row["metrics"]["pixel_mse_nearest_earlier"] == 0
        assert row["metrics"]["image_unchanged"] == 1
        assert row["metrics"]["children_identical_to_parent_fraction"] == 1
        assert row["metrics"]["distinct_child_images"] == 1
    assert rows[-1]["metrics"]["unchanged_image_streak"] == 3
    assert report["summary"]["metrics"]["image_unchanged"]["count"] == 2
    assert report["summary"]["metrics"]["image_unchanged"]["mean"] == 1
    assert report["summary"]["classifier_images"] == 0
    assert all("imagenet_selected_confidence" not in r["metrics"] for r in rows)
    assert report["final_image"] == rows[-1]["selected_image"]
    assert all(len(row["candidates"]) == 9 for row in rows)
    assert (tmp_path / "run" / report["final_image"]).is_file()
    assert (tmp_path / "run" / "metrics.csv").is_file()


@pytest.mark.parametrize("reference", ["previous-grid-mean", "previous-parent"])
def test_pixel_distances_and_novelty_match_saved_images_and_measurements(tmp_path, reference):
    directory = tmp_path / "run"
    report = run(directory, NoveltySelectionStrategy(novelty_reference=reference))
    session = json.loads((directory / "session.json").read_text())
    events = [e for e in session["events"] if e["action"] == "select"]
    previous = []
    for row, event in zip(report["generations"], events, strict=True):
        pixels = np.asarray(Image.open(directory / row["selected_image"]), dtype=float) / 255
        if previous:
            distances = [np.mean((pixels - earlier)**2) for earlier in previous]
            assert row["metrics"]["pixel_mse_previous"] == pytest.approx(distances[-1])
            assert row["metrics"]["pixel_mse_nearest_earlier"] == pytest.approx(min(distances))
            values = np.asarray(event["evaluation"]["values"])[:, 0]
            assert row["metrics"]["novelty_selected"] == values[event["position"]]
            assert row["metrics"]["novelty_grid_mean"] == pytest.approx(values.mean())
            if reference == "previous-parent":
                assert row["metrics"]["novelty_selected"] == pytest.approx(distances[-1])
                assert values[0] == 0  # The retained parent is its own novelty reference.
        else:
            assert row["metrics"]["novelty_selected"] is None
        previous.append(pixels)
    # Rebuilding is deterministic and never evaluates a model.
    assert build_run_metrics(directory) == report


def test_old_mean_offspring_records_keep_their_recorded_target():
    """The reader uses recorded targets without applying the new max default."""
    from automated_picbreeder.experiment_metrics import _align_offspring_feedback
    rows = [{"metrics": {"offspring_forecast_used": 1, "offspring_target": None}}]
    feedback = dict(origin_generation=0, received_generation=1, forecast=.4,
                    forecast_used=True, running_mean_forecast=.5, current_value=.6,
                    child_names=["fixed_reference_value"], child_values=[[0.]]*7+[[1.]],
                    target=.125, error=.275, absolute_error=.275, squared_error=.275**2,
                    running_mean_squared_error=.375**2, current_value_squared_error=.475**2)
    _align_offspring_feedback(rows, [feedback])
    assert rows[0]["metrics"]["offspring_target"] == .125


@pytest.mark.parametrize("weight", [0, .4, 1])
def test_combined_contributions_sum_to_recorded_score_including_initial_grid(tmp_path, weight):
    report = run(tmp_path / "run", NoveltyImageNetSelectionStrategy(weight, evaluator=Classifier()))
    for row in report["generations"]:
        m = row["metrics"]
        assert m["value_novelty_contribution"] + m["value_quality_contribution"] == pytest.approx(m["selected_score"])
        assert m["imagenet_grid_mean_confidence"] == pytest.approx(.7)
        assert len(row["candidates"]) == 9
        assert all(c["top_class"]["index"] == 0 for c in row["candidates"])
        assert row["selected_class"] == {"index": 0, "name": "same"}
    assert report["summary"]["classifier_images"] == 27
    assert report["generations"][0]["metrics"]["novelty_selected"] is None


def test_exploration_is_operation_not_difference_from_greedy(tmp_path):
    report = run(tmp_path / "run", ImageNetSelectionStrategy(1, evaluator=Classifier()), selection_seed=17)
    assert [r["selected_position"] for r in report["generations"]] == [Random(17).randrange(9), 6, 4]
    assert report["generations"][0]["selected_position"] == 8  # Random happened to choose greedy.
    for row in report["generations"]:
        assert row["metrics"]["exploratory_choice"] == 1
        assert row["metrics"]["exploratory_choice_frequency"] == 1


def test_batch_aggregation_weights_runs_equally_and_reports_missing_values():
    reports = [
        {"status": "complete", "generations": [{"generation": 0, "metrics": {"x": 1}}, {"generation": 1, "metrics": {"x": None}}],
         "summary": {"metrics": {"x": {"mean": 1., "final": None}}}},
        {"status": "complete", "generations": [{"generation": 0, "metrics": {"x": 9}}, {"generation": 1, "metrics": {"x": 9}}],
         "summary": {"metrics": {"x": {"mean": 9., "final": 9.}}}},
        {"status": "failed", "generations": [{"generation": 0, "metrics": {"x": 100}}],
         "summary": {"metrics": {"x": {"mean": 100., "final": 100.}}}},
    ]
    summary = summarize_batch(reports)
    assert summary["run_metrics"]["x"]["mean"]["mean"] == 5
    assert summary["run_metrics"]["x"]["final"]["count"] == 1
    assert summary["generations"][0]["metrics"]["x"]["median"] == 5
    assert summary["generations"][1]["metrics"]["x"]["count"] == 1
    assert summary["completed_runs"] == 2


def test_class_transition_uses_index_even_when_names_repeat(tmp_path):
    directory = tmp_path / "run"
    run(directory, ImageNetSelectionStrategy(evaluator=Classifier()))
    data = json.loads((directory / "session.json").read_text())
    events = [e for e in data["events"] if e["action"] == "select"]
    events[1]["evaluation"]["values"][events[1]["position"]] = [.1, .9]
    (directory / "session.json").write_text(json.dumps(data))
    rows = build_run_metrics(directory)["generations"]
    assert [r["metrics"]["imagenet_top_class_changed"] for r in rows] == [None, 1, 1]
    assert [r["selected_class"]["index"] for r in rows] == [0, 1, 0]


@pytest.mark.parametrize("name", ["novelty-predictability", "offspring-value", "offspring-value-imagenet"])
def test_learning_metrics_keep_preupdate_errors_and_align_delayed_targets(tmp_path, monkeypatch, name):
    pytest.importorskip("torch")
    from copy import deepcopy
    from automated_picbreeder.selection_strategies import (
        NoveltyPredictabilitySelectionStrategy, OffspringValueSelectionStrategy,
        OffspringValueImageNetSelectionStrategy,
    )

    class Observer:
        def __init__(self, **kwargs): self.updates = 0
        def initialize(self, seed): pass
        def describe(self): return {"observer": "controlled"}
        def state_hash(self): return str(self.updates)
        def prepare(self, image): return image.copy()
        def predict(self, images): return np.full(len(images), .5 / (self.updates + 1)), None
        def snapshot(self): return deepcopy(self)
        def train(self, replay, *, steps, batch_size, seed):
            self.updates += steps
            return {"updates": steps, "sampled_examples": steps * batch_size, "mean_loss": .123}

    class Predictor:
        def __init__(self, **kwargs): self.updates = 0
        def initialize(self, seed): pass
        def describe(self): return {"predictor": "controlled"}
        def state_hash(self): return str(self.updates)
        def prepare(self, images, mean, contexts): return list(zip(images, contexts))
        def predict(self, inputs): return np.linspace(0, 1, len(inputs)) / (self.updates + 1)
        def train(self, replay, *, steps, batch_size, seed):
            self.updates += steps
            return {"updates": steps, "sampled_examples": steps * batch_size, "mean_loss": .456}

    monkeypatch.setattr("automated_picbreeder.image_predictability.MaskedImageObserver", Observer)
    monkeypatch.setattr("automated_picbreeder.offspring_prediction.OffspringValuePredictor", Predictor)
    if name == "offspring-value-imagenet":
        strategy = OffspringValueImageNetSelectionStrategy(evaluator=Classifier(), warmup_targets=1, predictor_training_steps=1)
    elif name == "offspring-value":
        strategy = OffspringValueSelectionStrategy(comprehension_warmup_steps=2, training_steps=1,
                                                   warmup_targets=1, predictor_training_steps=1)
    else:
        strategy = NoveltyPredictabilitySelectionStrategy(comprehension_warmup_steps=2, training_steps=1)
    directory = tmp_path / "run"
    run_experiment(selection_strategy=strategy, output_dir=directory,
                   settings=ExperimentSettings(size=4, steps=6), progress=None)
    report = build_run_metrics(directory)
    rows = report["generations"]
    data = json.loads((directory / "session.json").read_text())
    events = [e for e in data["events"] if e["action"] == "select"]
    for i, row in enumerate(rows):
        m = row["metrics"]
        assert sum(row["candidates"][row["selected_position"]]["value_components"].values()) == pytest.approx(m["selected_score"])
        if name != "offspring-value-imagenet":
            assert m["prediction_mse_selected"] == (pytest.approx(.5/(i+1)) if i else None)
            assert m["value_quality_weight"] == (0 if i < 2 else .5)
            assert m["observer_training_loss"] == .123
        if name.startswith("offspring-value"):
            feedback = events[i]["evaluation"]["metadata"]["offspring"]["feedback"]
            if feedback is not None:
                origin = rows[feedback["origin_generation"]]
                assert origin["metrics"]["offspring_target"] == feedback["target"]
                assert origin["metrics"]["offspring_forecast_selected"] == feedback["forecast"]
                assert origin["metrics"]["offspring_error"] == feedback["forecast"] - feedback["target"]
                assert origin["offspring_outcome"]["received_generation"] == i
                assert len(origin["offspring_outcome"]["child_values"]) == 8
    if name.startswith("offspring-value"):
        eligible = 4 if name.endswith("imagenet") else 3
        assert rows[0]["metrics"]["offspring_target"] is None
        assert rows[-1]["metrics"]["offspring_target"] is None
        assert report["summary"]["offspring_prediction"]["all"]["count"] == eligible
        assert report["summary"]["offspring_prediction"]["forecast_used"]["count"] == eligible - 1
        errors = [r["metrics"]["offspring_error"] for r in rows if r["metrics"]["offspring_error"] is not None]
        assert report["summary"]["offspring_prediction"]["all"]["rmse"] == pytest.approx(np.sqrt(np.mean(np.square(errors))))
    assert build_run_metrics(directory) == report
