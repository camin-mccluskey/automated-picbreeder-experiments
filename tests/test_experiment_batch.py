import json

import pytest

from automated_picbreeder.experiment import ExperimentSettings, run_experiment
from automated_picbreeder.experiment_batch import run_batch
from automated_picbreeder.selection_strategies import NoveltySelectionStrategy


def read(path):
    return json.loads(path.read_text())


def test_batch_seeds_fresh_state_and_single_run_parity(tmp_path):
    built = []

    def factory():
        strategy = NoveltySelectionStrategy()
        built.append(strategy)
        return strategy

    settings = ExperimentSettings(seed=12, steps=3, size=4)
    output = tmp_path / "batch"
    manifest = run_batch(strategy_factory=factory, output_dir=output, runs=2, settings=settings, progress=None)
    assert len(built) == 2 and built[0] is not built[1]
    assert manifest["status"] == "complete"
    assert [r["seed"] for r in manifest["runs"]] == [12, 13]
    assert len({r["selection_seed"] for r in manifest["runs"]}) == 2
    for entry in manifest["runs"]:
        run_dir = output / entry["directory"]
        data = read(run_dir / "session.json")
        assert data["seed"] == entry["seed"]
        assert (output / entry["metrics"]).is_file()
        assert (output / entry["final_image"]).is_file()
        assert read(run_dir / "metrics.json")["generations"][0]["metrics"]["novelty_selected"] is None
    run_experiment(selection_strategy=NoveltySelectionStrategy(), output_dir=tmp_path / "single", settings=settings, progress=None)
    single = read(tmp_path / "single/session.json")
    batched = read(output / manifest["runs"][0]["directory"] / "session.json")
    assert single["events"] == batched["events"]
    assert single["genomes"] == batched["genomes"]
    assert read(output / "batch.json") == manifest
    assert read(output / "aggregate.json")["completed_runs"] == 2
    assert (output / "runs.csv").is_file()


def test_batch_increments_explicit_selection_seed_and_preserves_existing_output(tmp_path):
    output = tmp_path / "batch"
    manifest = run_batch(strategy_factory=NoveltySelectionStrategy, output_dir=output, runs=2,
                         settings=ExperimentSettings(seed=-2, selection_seed=0, steps=1, size=4), progress=None)
    assert [r["selection_seed"] for r in manifest["runs"]] == [0, 1]
    before = (output / "batch.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_batch(strategy_factory=lambda: pytest.fail("must not construct"), output_dir=output, runs=2)
    assert (output / "batch.json").read_bytes() == before


def test_failed_run_is_recorded_and_next_seed_still_runs(tmp_path):
    calls = 0

    class Failing(NoveltySelectionStrategy):
        def choose(self, images, *, rng):
            if self._generation == 1:
                raise RuntimeError("controlled failure")
            return super().choose(images, rng=rng)

    def factory():
        nonlocal calls
        calls += 1
        return Failing() if calls == 1 else NoveltySelectionStrategy()

    manifest = run_batch(strategy_factory=factory, output_dir=tmp_path / "batch", runs=2,
                         settings=ExperimentSettings(steps=3, size=4), progress=None)
    assert manifest["status"] == "completed_with_failures"
    assert [r["status"] for r in manifest["runs"]] == ["failed", "complete"]
    first = manifest["runs"][0]
    assert first["error"]["type"] == "RuntimeError"
    report = read(tmp_path / "batch" / first["metrics"])
    assert report["status"] == "failed"
    assert len(report["generations"]) == 1
    aggregate = read(tmp_path / "batch/aggregate.json")
    assert aggregate["completed_runs"] == 1
    assert manifest["failed_runs"] == 1


def test_interrupt_is_saved_and_stops_batch(tmp_path):
    def interrupt():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        run_batch(strategy_factory=interrupt, output_dir=tmp_path / "batch", runs=2,
                  settings=ExperimentSettings(steps=1, size=4), progress=None)
    manifest = read(tmp_path / "batch/batch.json")
    assert manifest["status"] == "interrupted"
    assert [r["status"] for r in manifest["runs"]] == ["interrupted", "pending"]


@pytest.mark.parametrize("runs", [0, -1, True, 1.5])
def test_invalid_run_count_has_no_side_effects(tmp_path, runs):
    with pytest.raises(ValueError):
        run_batch(strategy_factory=NoveltySelectionStrategy, output_dir=tmp_path / "batch", runs=runs)
    assert not (tmp_path / "batch").exists()
