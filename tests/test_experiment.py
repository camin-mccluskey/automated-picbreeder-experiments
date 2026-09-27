import json

import numpy as np
import pytest

from automated_picbreeder.breeding import BreedingSession
from automated_picbreeder.cppn import genome_from_dict, genome_to_dict, load_config, png_bytes, render
from automated_picbreeder.evaluation import Evaluation
from automated_picbreeder.experiment import ExperimentSettings, run_experiment
from automated_picbreeder.notebook import CPPNPlayground
from automated_picbreeder.selection import MaximumClassConfidence


def read_session(path):
    return json.loads((path / "session.json").read_text())


def test_selector_compares_maximum_over_any_class_and_preserves_parent_on_ties():
    selector = MaximumClassConfidence()
    assert selector.select(Evaluation(np.array([[.7, .1], [.1, .8]]), ("dog", "butterfly"))) == 1
    assert selector.select(Evaluation(np.array([[.7, .1], [.1, .7]]), ("dog", "butterfly"))) == 0
    with pytest.raises(ValueError, match="empty"):
        selector.select(Evaluation(np.empty((0, 2)), ("a", "b")))


def test_shared_engine_matches_human_ui_with_reselection_and_branching():
    automatic = BreedingSession(seed=7)
    human = CPPNPlayground(seed=7, size=8)
    for position, strength, topology in ((3, .2, True), (0, .5, False), (6, .1, True)):
        assert automatic.candidates == human.candidates
        automatic.select(position)
        human.select_buttons[position].click()
        human.strength.value, human.topology.value = strength, topology
        automatic.evolve(strength, topology)
        human.evolve_button.click()
        assert automatic.candidates == human.candidates
        assert automatic.parents == human.parents
        assert [genome_to_dict(g) for g in automatic.genomes.values()] == [genome_to_dict(g) for g in human.genomes.values()]
        for key in automatic.candidates:
            assert png_bytes(render(automatic.genomes[key], automatic.config, 8)) == human.images[key]
    automatic.back()
    human.back_button.click()
    automatic.reset()
    human.reset_button.click()
    assert automatic.events == human.events
    assert automatic.candidates == human.candidates


class ScriptedEvaluator:
    """Choose a new class on step 1, then retain the parent on step 2."""

    def __init__(self):
        self.calls = 0

    def evaluate(self, images):
        assert len(images) == 9
        assert all(image.ndim == 3 and image.shape[2] == 3 for image in images)
        values = np.full((9, 2), .01)
        if self.calls == 0:
            values[4, 0] = .7
        elif self.calls == 1:
            values[0, 0] = .7
            values[2, 1] = .8
        else:
            values[0, 1] = .8
            values[1, 0] = .8  # Exact tie must keep the parent.
        self.calls += 1
        return Evaluation(values, ("dog", "butterfly"), {"evaluator": "scripted"})


def test_runner_logs_rejected_alternatives_and_replays_in_human_loop(tmp_path):
    output = tmp_path / "experiment"
    evaluator = ScriptedEvaluator()
    summary = run_experiment(evaluator, MaximumClassConfidence(), output,
                             ExperimentSettings(steps=3, size=8, checkpoint_every=2), progress=None)
    data = read_session(output)
    decisions = [event for event in data["events"] if event["action"] == "select"]
    candidates = data["genomes"]
    assert data["format_version"] == 2
    assert data["output_encoding"] == "HSB"
    assert data["output_mapping"] == "picbreeder_hsb_v1"
    assert data["image_mode"] == "RGB"
    assert data["summary"] == summary
    assert summary["evaluated_images"] == 27
    assert summary["unique_candidates"] == 25
    assert evaluator.calls == 3
    assert len(candidates) == 25
    assert [d["position"] for d in decisions] == [4, 2, 0]
    assert [d["evaluation"]["names"][int(np.argmax(d["evaluation"]["values"][d["position"]]))]
            for d in decisions] == ["dog", "butterfly", "butterfly"]
    assert decisions[1]["displayed"][0] == decisions[0]["genome"]
    assert decisions[2]["genome"] == decisions[1]["genome"]
    assert len(list((output / "checkpoints").glob("*.json"))) == 2
    checkpoint = json.loads((output / "checkpoints/000001.json").read_text())
    assert checkpoint.keys() == data.keys()
    assert len(checkpoint["genomes"]) == 17
    assert len([e for e in checkpoint["events"] if e["action"] == "select"]) == 2
    assert all((output / g["image"]).is_file() for g in checkpoint["genomes"])
    assert len(list((output / "grids").glob("*.png"))) == 3
    assert len(list((output / "images").glob("*.png"))) == 25
    assert (output / "source/src/automated_picbreeder/persistence.py").is_file()
    human = CPPNPlayground(seed=7, size=8, save_dir=tmp_path / "human")
    config = load_config(output / "cppn.cfg")
    for i, decision in enumerate(decisions):
        if i:
            human.evolve()
        assert human.candidates == decision["displayed"]
        assert np.asarray(decision["evaluation"]["values"]).shape == (9, 2)
        human.select(decision["position"])
    human_output = human.save()
    human_data = read_session(human_output)
    # One reader can use both formats; only selection metadata/scores differ.
    assert human_data.keys() == data.keys()
    for field in ("seed", "size", "rendering", "output_encoding", "output_mapping", "image_mode", "versions", "config", "initial_config",
                  "genomes", "displayed", "selected"):
        assert human_data[field] == data[field]
    assert human_data["metadata"].keys() == data["metadata"].keys()
    assert human_data["summary"] is None
    for automatic, manual in zip(data["events"], human_data["events"], strict=True):
        assert automatic.keys() == manual.keys()
        assert {k: v for k, v in automatic.items() if k != "evaluation"} == {
            k: v for k, v in manual.items() if k != "evaluation"}
        if manual["action"] == "select":
            assert manual["evaluation"] is None
            assert automatic["evaluation"]["metadata"] == {"evaluator": "scripted"}
    for record in candidates:
        key = record["key"]
        assert genome_to_dict(genome_from_dict(record)) == genome_to_dict(human.genomes[key])
        image = (output / record["image"]).read_bytes()
        assert png_bytes(render(genome_from_dict(record), config, 8)) == image
        assert (human_output / record["image"]).read_bytes() == image
    assert (output / "selected.png").read_bytes() == human.images[human.selected_id]
    assert not (output / "candidates.jsonl").exists()
    assert not (output / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        run_experiment(evaluator, MaximumClassConfidence(), output, progress=None)


def test_one_decision_does_not_generate_unused_offspring(tmp_path):
    summary = run_experiment(ScriptedEvaluator(), MaximumClassConfidence(), tmp_path / "one",
                             ExperimentSettings(steps=1, size=8), progress=None)
    assert summary["unique_candidates"] == summary["evaluated_images"] == 9


def test_repeated_runs_preserve_candidate_and_decision_records(tmp_path):
    for name in ("a", "b"):
        run_experiment(ScriptedEvaluator(), MaximumClassConfidence(), tmp_path / name,
                       ExperimentSettings(steps=3, size=8), progress=None)
    a, b = read_session(tmp_path / "a"), read_session(tmp_path / "b")
    assert a["genomes"] == b["genomes"]
    assert a["events"] == b["events"]


def test_failed_evaluation_preserves_current_grid_and_previous_decisions(tmp_path):
    class FailingEvaluator(ScriptedEvaluator):
        def evaluate(self, images):
            if self.calls == 1:
                raise RuntimeError("evaluation failed")
            return super().evaluate(images)

    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="evaluation failed"):
        run_experiment(FailingEvaluator(), MaximumClassConfidence(), output,
                       ExperimentSettings(steps=3, size=8), progress=None)
    data = read_session(output)
    assert len(data["genomes"]) == 17
    assert len([e for e in data["events"] if e["action"] == "select"]) == 1
    assert data["events"][-1]["action"] == "evolve"
    assert all((output / g["image"]).is_file() for g in data["genomes"])
    assert data["summary"] is None


def test_evaluation_records_are_per_decision_and_detached_from_evaluator():
    session = BreedingSession()
    values = np.full((9, 1), .1)
    evaluation = Evaluation(values, ("novelty",), {"context": {"round": 0}})
    session.select(0, evaluation=evaluation)
    evaluation.values[:] = .9
    evaluation.metadata["context"]["round"] = 1
    session.select(0, evaluation=evaluation)
    first, second = session.events[-2:]
    assert first["genome"] == second["genome"]
    assert first["evaluation"]["values"][0] == [.1]
    assert second["evaluation"]["values"][0] == [.9]
    assert first["evaluation"]["metadata"]["context"]["round"] == 0
    with pytest.raises(ValueError, match="one row"):
        session.select(0, evaluation=Evaluation(np.ones((2, 1)), ("score",)))
