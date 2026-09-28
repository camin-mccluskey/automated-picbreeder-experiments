"""CLI configuration, dispatch and a real random run without Torch."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import Mock

import pytest

from automated_picbreeder.experiment import ExperimentSettings
from automated_picbreeder.selection_strategies import ImageNetSelectionStrategy, RandomSelectionStrategy


CLI = Path(__file__).resolve().parents[1] / "experiments" / "run_selection.py"


@pytest.mark.parametrize("name,epsilon", [("random", None), ("novelty", None), ("novelty-predictability", None), ("offspring-value", None), ("imagenet", None), ("imagenet", .2)])
def test_cli_constructs_selection_strategy_and_matches_python_defaults(tmp_path, monkeypatch, name, epsilon):
    if name in ("novelty-predictability", "offspring-value"):
        pytest.importorskip("torch")
    main = runpy.run_path(str(CLI))["main"]
    runner = Mock(return_value={"decisions": 100})
    monkeypatch.setitem(main.__globals__, "run_experiment", runner)
    factory = Mock()
    monkeypatch.setattr("automated_picbreeder.imagenet.ImageNetEvaluator", factory)
    output = tmp_path / "new"
    args = ["--selection-strategy", name, "--output", str(output)]
    if epsilon is not None:
        args += ["--epsilon", str(epsilon)]
    main(args)
    kwargs = runner.call_args.kwargs
    assert kwargs["settings"] == ExperimentSettings()
    assert kwargs["output_dir"] == output
    strategy = kwargs["selection_strategy"]
    if name == "random":
        assert isinstance(strategy, RandomSelectionStrategy)
        factory.assert_not_called()
    elif name == "novelty":
        assert strategy.describe()["selection_strategy"] == "novelty"
        factory.assert_not_called()
    elif name in ("novelty-predictability", "offspring-value"):
        assert strategy.describe()["selection_strategy"] == name
        assert strategy.comprehension_weight == .5
        assert strategy.observer.model is None
        factory.assert_not_called()
    else:
        assert isinstance(strategy, ImageNetSelectionStrategy)
        assert strategy.epsilon == (epsilon or 0)
        assert strategy.evaluator is factory.return_value
        assert factory.call_args.kwargs["device"] == "cpu"


@pytest.mark.parametrize("args", [
    ["--selection-strategy", "offspring-value", "--gamma", "nan"],
    ["--selection-strategy", "offspring-value", "--warmup-targets", "0"],
    ["--selection-strategy", "offspring-value", "--predictor-training-steps", "0"],
    ["--selection-strategy", "offspring-value", "--observer-initialization", "imagenet"],
    ["--selection-strategy", "novelty-predictability", "--gamma", "1"],
    ["--selection-strategy", "random", "--epsilon", "0"],
    ["--selection-strategy", "random", "--device", "cpu"],
    ["--selection-strategy", "novelty", "--epsilon", "0"],
    ["--selection-strategy", "novelty", "--device", "cpu"],
    ["--selection-strategy", "novelty", "--comprehension-weight", "0.5"],
    ["--selection-strategy", "novelty-predictability", "--comprehension-weight", "nan"],
    ["--selection-strategy", "novelty-predictability", "--observer-batch-size", "1"],
    ["--selection-strategy", "novelty-predictability", "--training-steps", "0"],
    ["--selection-strategy", "novelty-predictability", "--epsilon", "0"],
    ["--selection-strategy", "imagenet", "--epsilon", "-0.1"],
    ["--selection-strategy", "imagenet", "--epsilon", "1.1"],
    ["--selection-strategy", "imagenet", "--epsilon", "nan"],
    ["--selection-strategy", "imagenet", "--steps", "0"],
    ["--selection-strategy", "imagenet", "--size", "1"],
    ["--selection-strategy", "imagenet", "--mutation-strength", "nan"],
])
def test_invalid_cli_configuration_fails_before_model_or_output_creation(tmp_path, monkeypatch, args):
    main = runpy.run_path(str(CLI))["main"]
    factory = Mock(side_effect=AssertionError("Must not load a classifier"))
    monkeypatch.setattr("automated_picbreeder.imagenet.ImageNetEvaluator", factory)
    output = tmp_path / "new"
    with pytest.raises(SystemExit) as failure:
        main([*args, "--output", str(output)])
    assert failure.value.code == 2
    assert not output.exists()
    factory.assert_not_called()


def test_existing_output_is_rejected_before_classifier_loading(tmp_path, monkeypatch):
    main = runpy.run_path(str(CLI))["main"]
    factory = Mock(side_effect=AssertionError("Must not load a classifier"))
    monkeypatch.setattr("automated_picbreeder.imagenet.ImageNetEvaluator", factory)
    with pytest.raises(SystemExit) as failure:
        main(["--selection-strategy", "imagenet", "--output", str(tmp_path)])
    assert failure.value.code == 2
    assert list(tmp_path.iterdir()) == []
    factory.assert_not_called()


def test_observer_cli_passes_explicit_device(tmp_path, monkeypatch):
    main = runpy.run_path(str(CLI))["main"]
    factory = Mock()
    runner = Mock(return_value={"decisions": 2})
    monkeypatch.setitem(main.__globals__, "NoveltyPredictabilitySelectionStrategy", factory)
    monkeypatch.setitem(main.__globals__, "run_experiment", runner)
    main(["--selection-strategy", "novelty-predictability", "--device", "mps", "--output", str(tmp_path / "run")])
    assert factory.call_args.kwargs['device'] == 'mps'
    assert runner.call_args.kwargs['selection_strategy'] is factory.return_value


@pytest.mark.parametrize("strategy_name", ["random", "novelty"])
def test_cli_saves_complete_run_with_torch_unavailable(tmp_path, strategy_name):
    code = """
import builtins
import runpy
import sys
original_import = builtins.__import__
def without_torch(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision'}:
        raise ImportError('Torch deliberately unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_torch
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
assert 'torch' not in sys.modules
"""
    output = tmp_path / strategy_name
    result = subprocess.run([
        sys.executable, "-c", code, str(CLI), "--selection-strategy", strategy_name,
        "--steps", "2", "--size", "8", "--selection-seed", "0", "--output", str(output),
    ], capture_output=True, text=True, check=True)
    assert "Saved 2 decisions" in result.stdout
    data = json.loads((output / "session.json").read_text())
    assert data["summary"]["candidate_presentations"] == 18
    assert data["summary"]["evaluated_images"] == (0 if strategy_name == "random" else 18)
    assert data["metadata"]["settings"]["selection_seed"] == 0
    assert len(list((output / "images").glob("*.png"))) == 17
    assert len(list((output / "grids").glob("*.png"))) == 2
    assert (output / "source/experiments/run_selection.py").is_file()
    assert not (output / "source/experiments/imagenet_selection.py").exists()
    decisions = [e for e in data["events"] if e["action"] == "select"]
    if strategy_name == "random":
        assert all(e["evaluation"] is None and e["decision"] == {"mode": "random", "scores": None} for e in decisions)
    else:
        assert [e["decision"]["mode"] for e in decisions] == ["random", "greedy"]
        assert all(e["evaluation"]["names"] == ["pixel_novelty"] for e in decisions)
