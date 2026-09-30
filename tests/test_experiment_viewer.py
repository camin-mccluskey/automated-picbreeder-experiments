import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote

import pytest

from automated_picbreeder.experiment import ExperimentSettings, run_experiment
from automated_picbreeder.experiment_batch import run_batch
from automated_picbreeder.experiment_viewer import write_viewer
from automated_picbreeder.notebook import CPPNPlayground
from automated_picbreeder.selection_strategies import NoveltySelectionStrategy, RandomSelectionStrategy


def payload(path):
    html = path.read_text()
    return json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S)[1])


def test_single_run_viewer_is_offline_and_preserves_metrics(tmp_path):
    directory = tmp_path / "a run #1"
    run_experiment(selection_strategy=NoveltySelectionStrategy(), output_dir=directory,
                   settings=ExperimentSettings(steps=3, size=4), progress=None)
    html = directory / "index.html"
    assert html.is_file()
    report = payload(html)
    run = report["runs"][0]
    assert len(run["images"]) == 25
    assert len(run["report"]["generations"]) == 3
    assert run["report"]["generations"][0]["metrics"]["novelty_selected"] is None
    assert all((html.parent / unquote(image["url"])).is_file() for image in run["images"])
    assert not re.search(r'<(?:script|link)[^>]*(?:src|href)=["\']https?://', html.read_text())
    original = (directory / "session.json").read_bytes()
    moved = tmp_path / "reports" / "collection.html"
    write_viewer(directory, output=moved)
    run = payload(moved)["runs"][0]
    assert all((moved.parent / unquote(image["url"])).is_file() for image in run["images"])
    assert (directory / "session.json").read_bytes() == original
    assert (directory / "source/src/automated_picbreeder/viewer/report.js").is_file()


def test_collection_includes_manual_final_state_and_automated_batch(tmp_path):
    batch = tmp_path / "batch"
    run_batch(strategy_factory=RandomSelectionStrategy, output_dir=batch, runs=2,
              settings=ExperimentSettings(steps=2, size=4), progress=None)
    human = CPPNPlayground(size=4, save_dir=tmp_path / "manual")
    human.select(1)
    root = human.selected_id
    human.evolve()
    human.select(2)
    human.back()
    saved = human.save()
    assert (saved / "index.html").is_file()
    path = write_viewer(tmp_path, output=tmp_path / "index.html")
    data = payload(path)
    assert len(data["runs"]) == 3  # Batch runs are not discovered a second time.
    manual = next(r for r in data["runs"] if r["report"]["selection_strategy"]["selection_strategy"] == "human")
    assert manual["report"]["final_selected_id"] == root
    assert manual["report"]["generations"][-1]["selected_id"] != root
    assert len(manual["report"]["display_history"]) == 3
    assert len(manual["images"]) == 17
    group = next(g for g in data["groups"] if g["aggregate"] is not None)
    assert group["aggregate"]["completed_runs"] == 2


def test_batch_viewer_retains_failures_and_empty_manual_sessions(tmp_path):
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("fixture failure")
        return RandomSelectionStrategy()

    run_batch(strategy_factory=factory, output_dir=tmp_path / "batch", runs=2,
              settings=ExperimentSettings(steps=1, size=4), progress=None)
    data = payload(tmp_path / "batch/index.html")
    assert [r["status"] for r in data["runs"]] == ["failed", "complete"]
    assert data["runs"][0]["error"] == "RuntimeError: fixture failure"
    saved = CPPNPlayground(size=4, save_dir=tmp_path).save()
    manual = payload(saved / "index.html")["runs"][0]
    assert manual["report"]["generations"] == []
    assert manual["report"]["final_image"] is None
    assert len(manual["images"]) == 9


def test_embedded_data_cannot_close_script_and_paths_cannot_escape_run(tmp_path):
    directory = tmp_path / "run"
    run_experiment(selection_strategy=RandomSelectionStrategy(), output_dir=directory,
                   settings=ExperimentSettings(steps=1, size=4), progress=None)
    metrics = json.loads((directory / "metrics.json").read_text())
    attack = '</script><script>window.injected=true</script>'
    metrics["settings"]["note"] = attack
    (directory / "metrics.json").write_text(json.dumps(metrics))
    write_viewer(directory, refresh=False)
    html = (directory / "index.html").read_text()
    assert attack not in html
    assert payload(directory / "index.html")["runs"][0]["report"]["settings"]["note"] == attack
    data = json.loads((directory / "session.json").read_text())
    data["genomes"][0]["image"] = "../outside.png"
    (directory / "session.json").write_text(json.dumps(data))
    write_viewer(directory, refresh=False)
    assert "outside" in payload(directory / "index.html")["runs"][0]["error"]


def test_missing_source_and_unsafe_output_fail_before_writing(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_viewer(tmp_path / "missing")
    with pytest.raises(ValueError, match="html"):
        write_viewer(tmp_path, output=tmp_path / "session.json")


def test_images_remain_available_when_metrics_cannot_be_read(tmp_path):
    directory = tmp_path / "run"
    run_experiment(selection_strategy=RandomSelectionStrategy(), output_dir=directory,
                   settings=ExperimentSettings(steps=1, size=4), progress=None)
    (directory / "metrics.json").write_text("invalid json")
    run = payload(write_viewer(directory, refresh=False))["runs"][0]
    assert run["report"] is None
    assert run["error"]
    assert len(run["images"]) == 9
    assert "session.json" in run["files"]


def test_viewer_javascript_interactions(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the JavaScript DOM unit harness")
    run_batch(strategy_factory=NoveltySelectionStrategy, output_dir=tmp_path / "batch", runs=2,
              settings=ExperimentSettings(steps=3, size=4), progress=None)
    ui = CPPNPlayground(size=4, save_dir=tmp_path / "manual")
    ui.select(1)
    ui.evolve()
    ui.select(2)
    ui.back()
    ui.save()
    CPPNPlayground(size=4, save_dir=tmp_path / "empty").save()
    html = write_viewer(tmp_path)
    result = subprocess.run([node, str(Path(__file__).with_name("viewer_dom_test.cjs")), str(html)],
                            check=True, capture_output=True, text=True)
    assert "interaction unit checks passed" in result.stdout
