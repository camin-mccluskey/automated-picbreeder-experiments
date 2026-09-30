"""CLI configuration, dispatch and a real random run without Torch."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import Mock

import pytest

from automated_picbreeder.experiment import ExperimentSettings
from automated_picbreeder.selection_strategies import (
    ImageNetSelectionStrategy, NoveltyImageNetSelectionStrategy, RandomSelectionStrategy,
    OffspringValueImageNetSelectionStrategy,
)


CLI = Path(__file__).resolve().parents[1] / "experiments" / "run_selection.py"


@pytest.mark.parametrize("name,epsilon", [("random", None), ("novelty", None), ("novelty-imagenet", None), ("novelty-predictability", None), ("offspring-value", None), ("offspring-value-imagenet", None), ("imagenet", None), ("imagenet", .2)])
def test_cli_constructs_selection_strategy_and_matches_python_defaults(tmp_path, monkeypatch, name, epsilon):
    if name in ("novelty-predictability", "offspring-value", "offspring-value-imagenet"):
        pytest.importorskip("torch")
    main = runpy.run_path(str(CLI))["main"]
    runner = Mock(return_value={"decisions": 100})
    monkeypatch.setitem(main.__globals__, "run_experiment", runner)
    factory = Mock()
    monkeypatch.setattr("automated_picbreeder.imagenet.ImageNetEvaluator", factory)
    output = tmp_path / "new"
    args = [name, "--output", str(output)]
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
        assert strategy.comprehension_warmup_steps == 10
        assert strategy.observer.model is None
        factory.assert_not_called()
    else:
        if name == "offspring-value-imagenet":
            assert isinstance(strategy, OffspringValueImageNetSelectionStrategy)
            assert strategy.comprehension_weight == .5
            assert strategy.gamma == 1
            assert strategy.warmup_targets == 10
            assert strategy.predictor.model is None
        elif name == "novelty-imagenet":
            assert isinstance(strategy, NoveltyImageNetSelectionStrategy)
            assert strategy.comprehension_weight == .5
        else:
            assert isinstance(strategy, ImageNetSelectionStrategy)
            assert strategy.epsilon == (epsilon or 0)
        assert strategy.evaluator is factory.return_value
        assert factory.call_args.kwargs["device"] == "cpu"


@pytest.mark.parametrize("args", [
    ["offspring-value-imagenet", "--epsilon", "0"],
    ["offspring-value-imagenet", "--observer-initialization", "imagenet"],
    ["offspring-value-imagenet", "--comprehension-warmup-steps", "0"],
    ["offspring-value-imagenet", "--training-steps", "1"],
    ["offspring-value-imagenet", "--observer-batch-size", "2"],
    ["offspring-value-imagenet", "--learning-rate", "0.01"],
    ["offspring-value-imagenet", "--gamma", "nan"],
    ["offspring-value-imagenet", "--warmup-targets", "0"],
    ["offspring-value-imagenet", "--predictor-training-steps", "0"],
    ["offspring-value-imagenet", "--predictor-batch-size", "0"],
    ["offspring-value-imagenet", "--predictor-learning-rate", "inf"],
    ["offspring-value-imagenet", "--comprehension-weight", "nan"],
    ["offspring-value-imagenet", "--device", "cuda"],
    ["novelty-imagenet", "--epsilon", "0"],
    ["novelty-imagenet", "--comprehension-weight", "nan"],
    ["novelty-imagenet", "--comprehension-weight", "1.1"],
    ["novelty-imagenet", "--comprehension-warmup-steps", "0"],
    ["novelty-imagenet", "--observer-initialization", "imagenet"],
    ["novelty-imagenet", "--training-steps", "1"],
    ["novelty-imagenet", "--gamma", "0"],
    ["novelty-imagenet", "--imagenet-weights", "DEFAULT"],
    ["novelty-imagenet", "--imagenet-batch-size", "0"],
    ["imagenet", "--comprehension-weight", "0.5"],
    ["imagenet", "--imagenet-batch-size", "0"],
    ["imagenet", "--imagenet-weights", "DEFAULT"],
    ["random", "--cache-dir", "weights"],
    ["novelty", "--cache-dir", "weights"],
    ["novelty-predictability", "--imagenet-model", "resnet50"],
    ["offspring-value", "--imagenet-weights", "IMAGENET1K_V2"],
    ["random", "--imagenet-batch-size", "4"],
    ["offspring-value", "--gamma", "nan"],
    ["offspring-value", "--warmup-targets", "0"],
    ["offspring-value", "--predictor-training-steps", "0"],
    ["offspring-value", "--observer-initialization", "imagenet"],
    ["novelty-predictability", "--gamma", "1"],
    ["random", "--epsilon", "0"],
    ["random", "--device", "cpu"],
    ["novelty", "--epsilon", "0"],
    ["novelty", "--device", "cpu"],
    ["novelty", "--comprehension-weight", "0.5"],
    ["novelty-predictability", "--comprehension-weight", "nan"],
    ["novelty-predictability", "--observer-batch-size", "1"],
    ["novelty-predictability", "--training-steps", "0"],
    ["novelty-predictability", "--comprehension-warmup-steps", "-1"],
    ["offspring-value", "--comprehension-warmup-steps", "-1"],
    ["novelty", "--comprehension-warmup-steps", "10"],
    ["random", "--comprehension-warmup-steps", "10"],
    ["imagenet", "--comprehension-warmup-steps", "10"],
    ["novelty-predictability", "--epsilon", "0"],
    ["imagenet", "--epsilon", "-0.1"],
    ["imagenet", "--epsilon", "1.1"],
    ["imagenet", "--epsilon", "nan"],
    ["imagenet", "--steps", "0"],
    ["imagenet", "--size", "1"],
    ["imagenet", "--mutation-strength", "nan"],
])
def test_invalid_cli_configuration_fails_before_model_or_output_creation(tmp_path, monkeypatch, args):
    main = runpy.run_path(str(CLI))["main"]
    factory = Mock(side_effect=AssertionError("Must not load a classifier"))
    monkeypatch.setattr("automated_picbreeder.imagenet.ImageNetEvaluator", factory)
    for name in ("NoveltyPredictabilitySelectionStrategy", "OffspringValueSelectionStrategy",
                 "OffspringValueImageNetSelectionStrategy"):
        monkeypatch.setitem(main.__globals__, name, factory)
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
        main(["imagenet", "--output", str(tmp_path)])
    assert failure.value.code == 2
    assert list(tmp_path.iterdir()) == []
    factory.assert_not_called()


def test_observer_cli_passes_explicit_device(tmp_path, monkeypatch):
    main = runpy.run_path(str(CLI))["main"]
    factory = Mock()
    runner = Mock(return_value={"decisions": 2})
    monkeypatch.setitem(main.__globals__, "NoveltyPredictabilitySelectionStrategy", factory)
    monkeypatch.setitem(main.__globals__, "run_experiment", runner)
    main(["novelty-predictability", "--device", "mps", "--output", str(tmp_path / "run")])
    assert factory.call_args.kwargs['device'] == 'mps'
    assert runner.call_args.kwargs['selection_strategy'] is factory.return_value


@pytest.mark.parametrize("name,factory_name", [
    ("novelty-predictability", "NoveltyPredictabilitySelectionStrategy"),
    ("offspring-value", "OffspringValueSelectionStrategy"),
])
def test_cli_forwards_all_observer_predictor_and_run_options(tmp_path, monkeypatch, name, factory_name):
    main = runpy.run_path(str(CLI))["main"]
    factory = Mock()
    runner = Mock(return_value={"decisions": 12})
    monkeypatch.setitem(main.__globals__, factory_name, factory)
    monkeypatch.setitem(main.__globals__, "run_experiment", runner)
    initialization = "imagenet" if name == "novelty-predictability" else "random"
    output, cache = tmp_path / "run", tmp_path / "weights"
    args = [
        name, "--output", str(output), "--cache-dir", str(cache),
        "--device", "mps", "--comprehension-weight", "0.7",
        "--comprehension-warmup-steps", "0",
        "--observer-initialization", initialization, "--training-steps", "3",
        "--observer-batch-size", "4", "--learning-rate", "0.002",
        "--seed", "9", "--selection-seed", "42", "--steps", "12", "--size", "32",
        "--mutation-strength", "0.4", "--no-topology", "--checkpoint-every", "3",
    ]
    expected = dict(comprehension_weight=.7, comprehension_warmup_steps=0, observer_initialization=initialization,
                    training_steps=3, batch_size=4, learning_rate=.002,
                    device="mps", cache_dir=cache)
    if name == "offspring-value":
        args += ["--gamma", "0.3", "--warmup-targets", "2", "--predictor-training-steps", "5",
                 "--predictor-batch-size", "6", "--predictor-learning-rate", "0.003"]
        expected.update(gamma=.3, warmup_targets=2, predictor_training_steps=5,
                        predictor_batch_size=6, predictor_learning_rate=.003)
    main(args)
    factory.assert_called_once_with(**expected)
    runner.assert_called_once_with(
        selection_strategy=factory.return_value, output_dir=output,
        settings=ExperimentSettings(seed=9, selection_seed=42, steps=12, size=32,
                                    mutation_strength=.4, topology=False, checkpoint_every=3),
    )


@pytest.mark.parametrize("name", ["imagenet", "novelty-imagenet"])
def test_cli_forwards_all_imagenet_options(tmp_path, monkeypatch, name):
    main = runpy.run_path(str(CLI))["main"]
    factory = Mock()
    runner = Mock(return_value={"decisions": 100})
    monkeypatch.setattr("automated_picbreeder.imagenet.ImageNetEvaluator", factory)
    monkeypatch.setitem(main.__globals__, "run_experiment", runner)
    cache = tmp_path / "weights"
    strategy_options = ["--epsilon", "0.25"] if name == "imagenet" else ["--comprehension-weight", "0.7"]
    main([name, *strategy_options,
          "--imagenet-model", "resnet50", "--imagenet-weights", "IMAGENET1K_V2",
          "--imagenet-batch-size", "3", "--device", "cuda", "--cache-dir", str(cache),
          "--output", str(tmp_path / "run")])
    factory.assert_called_once_with(model_name="resnet50", weights="IMAGENET1K_V2",
                                    batch_size=3, device="cuda", cache_dir=cache)
    strategy = runner.call_args.kwargs["selection_strategy"]
    if name == "imagenet":
        assert strategy.epsilon == .25
    else:
        assert strategy.comprehension_weight == .7
    assert strategy.evaluator is factory.return_value


def test_cli_forwards_imagenet_offspring_options(tmp_path, monkeypatch):
    main = runpy.run_path(str(CLI))["main"]
    evaluator, strategy = Mock(), Mock()
    runner = Mock(return_value={"decisions": 14})
    monkeypatch.setattr("automated_picbreeder.imagenet.ImageNetEvaluator", evaluator)
    monkeypatch.setitem(main.__globals__, "OffspringValueImageNetSelectionStrategy", strategy)
    monkeypatch.setitem(main.__globals__, "run_experiment", runner)
    cache = tmp_path / "weights"
    main(["offspring-value-imagenet", "--comprehension-weight", "0.7",
          "--gamma", "2", "--warmup-targets", "3", "--predictor-training-steps", "4",
          "--predictor-batch-size", "5", "--predictor-learning-rate", "0.002",
          "--imagenet-model", "resnet50", "--imagenet-weights", "IMAGENET1K_V2",
          "--imagenet-batch-size", "3", "--device", "mps", "--cache-dir", str(cache),
          "--output", str(tmp_path / "run"), "--steps", "14"])
    evaluator.assert_called_once_with(model_name="resnet50", weights="IMAGENET1K_V2",
                                     batch_size=3, device="mps", cache_dir=cache)
    strategy.assert_called_once_with(comprehension_weight=.7, gamma=2, warmup_targets=3,
        predictor_training_steps=4, predictor_batch_size=5, predictor_learning_rate=.002,
        device="mps", evaluator=evaluator.return_value)
    assert runner.call_args.kwargs['selection_strategy'] is strategy.return_value


def test_invalid_imagenet_configuration_is_reported_as_cli_error(tmp_path, monkeypatch, capsys):
    main = runpy.run_path(str(CLI))["main"]
    factory = Mock(side_effect=ValueError("Unknown model 'invalid'"))
    runner = Mock()
    monkeypatch.setattr("automated_picbreeder.imagenet.ImageNetEvaluator", factory)
    monkeypatch.setitem(main.__globals__, "run_experiment", runner)
    output = tmp_path / "run"
    with pytest.raises(SystemExit) as failure:
        main(["imagenet", "--imagenet-model", "invalid", "--output", str(output)])
    assert failure.value.code == 2
    assert "Unknown model 'invalid'" in capsys.readouterr().err
    assert not output.exists()
    runner.assert_not_called()


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
        sys.executable, "-c", code, str(CLI), strategy_name,
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
    assert (output / "source/src/automated_picbreeder/selection_cli.py").is_file()
    assert not (output / "source/experiments/imagenet_selection.py").exists()
    decisions = [e for e in data["events"] if e["action"] == "select"]
    if strategy_name == "random":
        assert all(e["evaluation"] is None and e["decision"] == {"mode": "random", "scores": None} for e in decisions)
    else:
        assert [e["decision"]["mode"] for e in decisions] == ["random", "greedy"]
        assert all(e["evaluation"]["names"] == ["pixel_novelty"] for e in decisions)


@pytest.mark.parametrize("name,expected,absent", [
    ("random", ("Run settings:", "--steps"), ("--device", "--epsilon", "--gamma")),
    ("imagenet", ("ImageNet classifier:", "--epsilon"), ("--gamma", "--training-steps")),
    ("offspring-value-imagenet", ("Offspring predictor:", "--gamma", "--imagenet-model"),
     ("--epsilon", "--observer-initialization", "--comprehension-warmup-steps")),
])
def test_strategy_help_lists_only_applicable_options(name, expected, absent, capsys):
    main = runpy.run_path(str(CLI))["main"]
    with pytest.raises(SystemExit) as result:
        main([name, "--help"])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert all(text in help_text for text in expected)
    assert all(text not in help_text for text in absent)


@pytest.mark.parametrize("args", [[], ["--selection-strategy", "random"],
    ["unknown"], ["random", "--ste", "2"], ["imagenet", "--eps", "0.1"]])
def test_requires_known_subcommand_and_full_option_names(args):
    main = runpy.run_path(str(CLI))["main"]
    with pytest.raises(SystemExit) as result:
        main(args)
    assert result.value.code == 2


STRATEGIES = (
    "random", "novelty", "imagenet", "novelty-imagenet", "novelty-predictability",
    "offspring-value", "offspring-value-imagenet",
)
OPTION_FAMILIES = (
    ("--epsilon", "0", {"imagenet"}),
    ("--imagenet-model", "resnet18", {"imagenet", "novelty-imagenet", "offspring-value-imagenet"}),
    ("--comprehension-weight", "0.5", {"novelty-imagenet", "novelty-predictability", "offspring-value", "offspring-value-imagenet"}),
    ("--training-steps", "1", {"novelty-predictability", "offspring-value"}),
    ("--gamma", "1", {"offspring-value", "offspring-value-imagenet"}),
    ("--device", "cpu", set(STRATEGIES) - {"random", "novelty"}),
)


def test_batch_cli_runs_without_torch_and_records_seed_schedule(tmp_path):
    code = """
import builtins, runpy, sys
original_import = builtins.__import__
def without_torch(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision'}:
        raise ImportError('Torch deliberately unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_torch
runpy.run_path(sys.argv[1])['main'](['novelty', '--runs', '2', '--steps', '2', '--size', '4',
    '--seed', '20', '--selection-seed', '0', '--output', sys.argv[2]])
assert 'torch' not in sys.modules
"""
    output = tmp_path / "batch"
    result = subprocess.run([sys.executable, "-c", code, str(CLI), str(output)],
                            capture_output=True, text=True, check=True)
    assert "Saved 2/2 completed runs" in result.stdout
    manifest = json.loads((output / "batch.json").read_text())
    assert [r["seed"] for r in manifest["runs"]] == [20, 21]
    assert [r["selection_seed"] for r in manifest["runs"]] == [0, 1]


def test_failed_batch_exits_nonzero(tmp_path, monkeypatch):
    from automated_picbreeder import selection_cli
    monkeypatch.setattr(selection_cli, "run_batch", lambda **kwargs: {"completed_runs": 1, "failed_runs": 1})
    with pytest.raises(SystemExit) as result:
        selection_cli.main(["random", "--runs", "2", "--output", str(tmp_path / "batch")])
    assert result.value.code == 1


@pytest.mark.parametrize("count", ["0", "-1", "1.5"])
def test_invalid_batch_size_fails_before_output(tmp_path, count):
    from automated_picbreeder.selection_cli import main
    with pytest.raises(SystemExit) as result:
        main(["random", "--runs", count, "--output", str(tmp_path / "batch")])
    assert result.value.code == 2
    assert not (tmp_path / "batch").exists()


@pytest.mark.parametrize("name,flag,value", [
    (name, flag, value) for name in STRATEGIES
    for flag, value, supported in OPTION_FAMILIES if name not in supported
])
def test_unrelated_option_families_are_rejected_by_parser(name, flag, value, capsys):
    from automated_picbreeder.selection_cli import build_parser

    with pytest.raises(SystemExit) as result:
        build_parser().parse_args([name, flag, value])
    assert result.value.code == 2
    assert f"unrecognized arguments: {flag}" in capsys.readouterr().err


def test_all_help_works_without_torch_and_lists_strategies():
    code = '''
import builtins
import runpy
import sys
original_import = builtins.__import__
def without_torch(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision'}:
        raise ImportError('Torch deliberately unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_torch
main = runpy.run_path(sys.argv[1])['main']
for prefix in [[], *[[name] for name in sys.argv[2:]]]:
    try:
        main([*prefix, '--help'])
    except SystemExit as result:
        assert result.code == 0
    else:
        raise AssertionError('Help must exit')
assert 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", code, str(CLI), *STRATEGIES],
                            capture_output=True, text=True, check=True)
    top_level = result.stdout.split("Run settings:")[0]
    assert all(name in top_level for name in STRATEGIES)
