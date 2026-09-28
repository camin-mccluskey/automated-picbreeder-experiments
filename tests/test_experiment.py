import json
import random

import numpy as np
import pytest

from automated_picbreeder.breeding import BreedingSession
from automated_picbreeder.cppn import genome_from_dict, genome_to_dict, load_config, png_bytes, render
from automated_picbreeder.evaluation import Evaluation
from automated_picbreeder.experiment import ExperimentSettings, run_experiment
from automated_picbreeder.notebook import CPPNPlayground
from automated_picbreeder.interpretability import load_network
from automated_picbreeder.selection_strategies import (
    ImageNetSelectionStrategy, RandomSelectionStrategy, SelectionDecision,
)


def read_session(path):
    return json.loads((path / "session.json").read_text())


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

    def describe(self):
        return {"evaluator": "scripted"}

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
    summary = run_experiment(selection_strategy=ImageNetSelectionStrategy(evaluator=evaluator), output_dir=output,
                             settings=ExperimentSettings(steps=3, size=8, checkpoint_every=2), progress=None)
    data = read_session(output)
    decisions = [event for event in data["events"] if event["action"] == "select"]
    candidates = data["genomes"]
    assert data["format_version"] == 2
    assert data["output_encoding"] == "HSB"
    assert data["output_mapping"] == "picbreeder_hsb_v1"
    assert data["image_mode"] == "RGB"
    assert data["summary"] == summary
    assert summary["evaluated_images"] == summary["candidate_presentations"] == 27
    assert summary["selection_mode"] == "greedy"
    assert summary["selected_score"] == .8
    assert data["metadata"]["selection_strategy"]["selection_strategy"] == "imagenet"
    assert isinstance(data["metadata"]["settings"]["selection_seed"], int)
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
        assert {k: v for k, v in automatic.items() if k not in {"evaluation", "decision"}} == {
            k: v for k, v in manual.items() if k not in {"evaluation", "decision"}}
        if manual["action"] == "select":
            assert manual["evaluation"] is None
            assert manual["decision"] is None
            assert automatic["decision"]["mode"] == "greedy"
            np.testing.assert_allclose(automatic["decision"]["scores"],
                                       np.max(automatic["evaluation"]["values"], axis=1))
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
        run_experiment(selection_strategy=ImageNetSelectionStrategy(evaluator=evaluator), output_dir=output, progress=None)


def test_one_decision_does_not_generate_unused_offspring(tmp_path):
    summary = run_experiment(selection_strategy=ImageNetSelectionStrategy(evaluator=ScriptedEvaluator()), output_dir=tmp_path / "one",
                             settings=ExperimentSettings(steps=1, size=8), progress=None)
    assert summary["unique_candidates"] == summary["evaluated_images"] == 9


def test_repeated_runs_preserve_candidate_and_decision_records(tmp_path):
    for name in ("a", "b"):
        run_experiment(selection_strategy=ImageNetSelectionStrategy(evaluator=ScriptedEvaluator()), output_dir=tmp_path / name,
                       settings=ExperimentSettings(steps=3, size=8), progress=None)
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
        run_experiment(selection_strategy=ImageNetSelectionStrategy(evaluator=FailingEvaluator()), output_dir=output,
                       settings=ExperimentSettings(steps=3, size=8), progress=None)
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
    decision = SelectionDecision(position=0, mode="greedy", evaluation=evaluation, scores=evaluation.values[:, 0])
    session.select(0, decision=decision)
    evaluation.values[:] = .9
    evaluation.metadata["context"]["round"] = 1
    decision = SelectionDecision(position=0, mode="greedy", evaluation=evaluation, scores=evaluation.values[:, 0])
    session.select(0, decision=decision)
    first, second = session.events[-2:]
    assert first["genome"] == second["genome"]
    assert first["evaluation"]["values"][0] == [.1]
    assert second["evaluation"]["values"][0] == [.9]
    assert first["evaluation"]["metadata"]["context"]["round"] == 0
    assert first["decision"]["scores"][0] == .1
    decision.scores[:] = .5
    assert second["decision"]["scores"][0] == .9
    with pytest.raises(ValueError, match="one row"):
        session.select(0, decision=SelectionDecision(position=0, mode="greedy",
                       evaluation=Evaluation(np.ones((2, 1)), ("score",))))


class ConstantEvaluator:
    def evaluate(self, images):
        return Evaluation(np.tile([.1, .9], (len(images), 1)), ("a", "b"))

    def describe(self):
        return {"evaluator": "constant"}


@pytest.mark.parametrize("epsilon", [None, 0, .3, 1])
def test_all_selection_modes_reproduce_records_and_load_saved_networks(tmp_path, epsilon):
    strategy = (RandomSelectionStrategy() if epsilon is None else
                ImageNetSelectionStrategy(epsilon=epsilon, evaluator=ConstantEvaluator()))
    settings = ExperimentSettings(steps=3, size=8, selection_seed=19, checkpoint_every=2)
    global_rng = random.getstate()
    runs = []
    for name in ("first", "repeated"):
        output = tmp_path / name
        summary = run_experiment(selection_strategy=strategy, output_dir=output, settings=settings, progress=None)
        data = read_session(output)
        runs.append(data)
        assert summary["candidate_presentations"] == 27
        assert summary["evaluated_images"] == (0 if epsilon is None else 27)
        assert summary["unique_candidates"] == 25
        assert (summary["selected_score"] is None) == (epsilon is None)
        assert "selected_class_name" not in summary
        loaded = load_network(output)
        assert loaded.genome.key == summary["selected_id"]
        load_network(output / "checkpoints" / "000001.json")
        replay = BreedingSession(seed=settings.seed)
        for step, event in enumerate(e for e in data["events"] if e["action"] == "select"):
            if step:
                replay.evolve(settings.mutation_strength, settings.topology)
            assert replay.candidates == event["displayed"]
            replay.select(event["position"])
            assert (event["evaluation"] is None) == (epsilon is None)
        for record in data["genomes"]:
            assert genome_to_dict(replay.genomes[record["key"]]) == genome_to_dict(genome_from_dict(record))
            assert png_bytes(render(replay.genomes[record["key"]], replay.config, 8)) == (output / record["image"]).read_bytes()
    assert runs[0]["genomes"] == runs[1]["genomes"]
    assert runs[0]["events"] == runs[1]["events"]
    assert random.getstate() == global_rng


def test_recorded_selection_seed_replays_random_choices(tmp_path):
    for seed in (None, 0):
        output = tmp_path / str(seed)
        run_experiment(selection_strategy=RandomSelectionStrategy(), output_dir=output,
                       settings=ExperimentSettings(steps=4, size=4, selection_seed=seed), progress=None)
        data = read_session(output)
        resolved = data["metadata"]["settings"]["selection_seed"]
        assert isinstance(resolved, int)
        if seed is not None:
            assert resolved == seed
        rng = random.Random(resolved)
        assert [e["position"] for e in data["events"] if e["action"] == "select"] == [rng.randrange(9) for _ in range(4)]


def test_selection_rng_consumption_does_not_change_breeding(tmp_path):
    class FixedChoices:
        def __init__(self, draws):
            self.draws = draws

        def describe(self):
            return {"selection_strategy": "fixed_choices", "draws": self.draws}

        def choose(self, images, *, rng):
            for _ in range(self.draws):
                rng.random()
            return SelectionDecision(position=2, mode="fixed")

    for draws in (0, 100):
        run_experiment(selection_strategy=FixedChoices(draws), output_dir=tmp_path / str(draws),
                       settings=ExperimentSettings(steps=3, size=4), progress=None)
    a, b = read_session(tmp_path / "0"), read_session(tmp_path / "100")
    assert a["genomes"] == b["genomes"]
    assert a["events"] == b["events"]


def test_runner_uses_reported_preference_scores_not_largest_measurement(tmp_path):
    class MinimiseFirstMeasurement:
        def describe(self):
            return {"selection_strategy": "minimise_first_measurement"}

        def choose(self, images, *, rng):
            return SelectionDecision(position=0, mode="custom", scores=-np.arange(9.),
                                     evaluation=Evaluation(np.column_stack((np.arange(9.), np.full(9, 100))),
                                                           ("cost", "unrelated measurement")))

    progress = []
    summary = run_experiment(selection_strategy=MinimiseFirstMeasurement(), output_dir=tmp_path / "run",
                             settings=ExperimentSettings(steps=1, size=4), progress=progress.append)
    assert summary["selected_score"] == 0
    assert summary["selection_mode"] == "custom"
    assert "100.000000" not in progress[0]
    assert "custom" in progress[0]


@pytest.mark.parametrize("failure", ["raise", "position", "scores", "rows", "columns"])
def test_failed_selection_preserves_grid_and_prior_decisions(tmp_path, failure):
    class FailingStrategy:
        def __init__(self):
            self.calls = 0

        def describe(self):
            return {"selection_strategy": "failing"}

        def choose(self, images, *, rng):
            self.calls += 1
            if self.calls == 1:
                return SelectionDecision(position=1, mode="custom", evaluation=Evaluation(np.ones((9, 1)), ("original",)))
            if failure == "raise":
                raise ValueError("Selection failed")
            return SelectionDecision(
                position=9 if failure == "position" else 0, mode="custom",
                scores=np.ones(8) if failure == "scores" else None,
                evaluation=Evaluation(np.ones((8 if failure == "rows" else 9, 1)), ("changed",)),
            )

    output = tmp_path / failure
    with pytest.raises(ValueError):
        run_experiment(selection_strategy=FailingStrategy(), output_dir=output,
                       settings=ExperimentSettings(steps=3, size=4), progress=None)
    data = read_session(output)
    assert len(data["genomes"]) == 17
    assert len([e for e in data["events"] if e["action"] == "select"]) == 1
    assert data["events"][-1]["action"] == "evolve"
    assert data["summary"] is None
    assert all((output / record["image"]).is_file() for record in data["genomes"])


def test_session_rejects_decision_for_a_different_position():
    session = BreedingSession(7)
    original_events = len(session.events)
    with pytest.raises(ValueError, match="position"):
        session.select(0, decision=SelectionDecision(position=1, mode="random"))
    assert session.selected_id is None
    assert len(session.events) == original_events


@pytest.mark.parametrize("kwargs", [{"seed": True}, {"seed": 1.5}, {"selection_seed": False}, {"selection_seed": "7"}])
def test_settings_reject_ambiguous_seeds(kwargs):
    with pytest.raises(ValueError, match="seed"):
        ExperimentSettings(**kwargs)
