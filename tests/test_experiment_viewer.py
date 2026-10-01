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
    assert run["experiment"] == directory.name
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
    assert "generations" not in manual["report"]
    assert len(manual["images"]) == 1
    manual = payload(path.parent / unquote(manual["viewer_url"]))["runs"][0]
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
    session = json.loads((directory / "session.json").read_text())
    attack = '</script><script>window.injected=true</script>'
    session["metadata"]["settings"]["note"] = attack
    (directory / "session.json").write_text(json.dumps(session))
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
    (directory / "performance.json").write_text("invalid json")
    run = payload(write_viewer(directory, refresh=False))["runs"][0]
    assert run["report"] is None
    assert run["error"]
    assert len(run["images"]) == 9
    assert "session.json" in run["files"]


def test_viewer_javascript_interactions(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the JavaScript DOM unit harness")
    run_batch(strategy_factory=lambda: NoveltySelectionStrategy(novelty_reference="previous-parent"),
              output_dir=tmp_path / "batch", runs=2,
              settings=ExperimentSettings(steps=3, size=4), progress=None)
    run_batch(strategy_factory=lambda: NoveltySelectionStrategy(novelty_reference="previous-parent"),
              output_dir=tmp_path / "batch-repeat", runs=1,
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


def test_experiment_name_survives_collection_and_direct_run_views(tmp_path):
    for name in ("experiment-one", "experiment-two"):
        run_batch(strategy_factory=RandomSelectionStrategy, output_dir=tmp_path / name,
                  runs=1, settings=ExperimentSettings(steps=1, size=4), progress=None)
    html = write_viewer(tmp_path, output=tmp_path / "reports" / "collection.html")
    runs = payload(html)["runs"]
    assert [run["experiment"] for run in runs] == ["experiment-one", "experiment-two"]
    assert runs[0]["label"] == runs[1]["label"] == "run-0000-seed7"
    for run in runs:
        detail = payload(html.parent / unquote(run["viewer_url"]))
        assert detail["runs"][0]["experiment"] == run["experiment"]
        assert [item["experiment"] for item in detail["navigation"]] == ["experiment-one", "experiment-two"]
        directory = tmp_path / run["experiment"] / run["label"]
        direct = payload(write_viewer(directory))["runs"][0]
        assert direct["experiment"] == run["experiment"]
        assert direct["label"] == run["label"]


def test_nested_runs_with_identical_folder_names_have_distinct_titles(tmp_path):
    for name in ("first", "second"):
        run_experiment(selection_strategy=RandomSelectionStrategy(), output_dir=tmp_path / name / "run",
                       settings=ExperimentSettings(steps=1, size=4), progress=None)
    runs = payload(write_viewer(tmp_path))["runs"]
    assert [run["experiment"] for run in runs] == ["first/run", "second/run"]


def test_cached_viewer_avoids_session_parsing_and_png_loading(tmp_path, monkeypatch):
    from automated_picbreeder import viewer_cache
    from PIL import Image
    directory = tmp_path / "run"
    run_experiment(selection_strategy=RandomSelectionStrategy(), output_dir=directory,
                   settings=ExperimentSettings(steps=3, size=4), progress=None)
    before = (directory / 'session.json').read_bytes()
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        assert path.name != 'session.json', 'Warm viewer parsed the session'
        return original_read(path, *args, **kwargs)

    def unexpected(*args, **kwargs):
        pytest.fail('Warm viewer rebuilt diagnostics or loaded a PNG')

    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(viewer_cache, 'build_run_metrics', unexpected)
    monkeypatch.setattr(Image, 'open', unexpected)
    first = write_viewer(directory)
    stamp = first.stat().st_mtime_ns
    write_viewer(directory)
    assert first.stat().st_mtime_ns == stamp
    assert (directory / 'session.json').read_bytes() == before


@pytest.mark.parametrize('change', ['session', 'performance', 'posthoc', 'image', 'code', 'corrupt', 'missing', 'refresh'])
def test_cache_invalidation(tmp_path, monkeypatch, change):
    from automated_picbreeder import viewer_cache
    directory = tmp_path / 'run'
    run_experiment(selection_strategy=RandomSelectionStrategy(), output_dir=directory,
                   settings=ExperimentSettings(steps=2, size=4), progress=None)
    original = viewer_cache.build_run_metrics
    calls = []

    def build(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(viewer_cache, 'build_run_metrics', build)
    cache = directory / '.viewer-cache/records.json'
    if change in {'session', 'performance'}:
        path = directory / (change + '.json')
        data = json.loads(path.read_text())
        if change == 'session':
            data['metadata']['settings']['audit_note'] = 'updated'
        else:
            data['elapsed_seconds'] = 123.0
        path.write_text(json.dumps(data))
    elif change == 'posthoc':
        # A malformed newly added evaluation must not be masked by an old cache.
        (directory / 'posthoc_imagenet.json').write_text('{}')
    elif change == 'image':
        from PIL import Image
        selected = json.loads(cache.read_text())['report']['final_image']
        Image.new('RGB', (4, 4), (13, 27, 81)).save(directory / selected)
    elif change == 'code':
        monkeypatch.setattr(viewer_cache, 'code_version', lambda: 'changed implementation')
    elif change == 'corrupt':
        cache.write_text('{broken')
    elif change == 'missing':
        cache.unlink()
    result = payload(write_viewer(directory, refresh=change == 'refresh'))['runs'][0]
    assert calls == [True]
    if change == 'posthoc':
        assert result['error'] and result['report'] is None
    else:
        assert result['error'] is None
    if change == 'performance':
        assert result['report']['summary']['run_seconds'] == 123.0
    if change == 'session':
        assert result['report']['settings']['audit_note'] == 'updated'


def test_reindex_old_run_preserves_original_records(tmp_path):
    from automated_picbreeder.experiment_metrics import build_run_metrics
    directory = tmp_path / 'old-run'
    run_experiment(selection_strategy=RandomSelectionStrategy(), output_dir=directory,
                   settings=ExperimentSettings(steps=3, size=4), progress=None)
    (directory / '.viewer-cache/records.json').unlink()
    saved = json.loads((directory / 'metrics.json').read_text())
    saved.pop('image_inventory')
    (directory / 'metrics.json').write_text(json.dumps(saved))
    originals = {name: (directory / name).read_bytes() for name in ['session.json', 'metrics.json', 'metrics.csv']}
    report = payload(write_viewer(directory))['runs'][0]['report']
    expected = build_run_metrics(directory)
    expected.pop('image_inventory')
    assert report == expected
    assert all((directory / name).read_bytes() == content for name, content in originals.items())


def test_parallel_matches_serial_and_overview_links_work(tmp_path):
    batch = tmp_path / 'batch'
    run_batch(strategy_factory=RandomSelectionStrategy, output_dir=batch, runs=2,
              settings=ExperimentSettings(steps=3, size=4), progress=None)
    output = tmp_path / 'reports with spaces #1' / 'collection.html'
    write_viewer(batch, output=output, refresh=True, jobs=1)
    overview = payload(output)
    serial = {p: p.read_bytes() for p in output.parent.rglob('*.html')}
    write_viewer(batch, output=output, refresh=True, jobs=2)
    assert all(p.read_bytes() == content for p, content in serial.items())
    for run in overview['runs']:
        detail = output.parent / unquote(run['viewer_url'])
        page = payload(detail)
        assert (detail.parent / unquote(page['overview'])).resolve() == output.resolve()
        full = page['runs'][0]
        assert len(full['images']) == 25
        assert run['image_count'] == 25
        assert len(run['images']) == 1
        assert 'generations' not in run['report']
        assert run['series']['pixel_mse_previous'] == [g['metrics']['pixel_mse_previous'] for g in full['report']['generations']]
        assert all((detail.parent / unquote(image['url'])).is_file() for image in full['images'])


def test_save_pipeline_does_not_reextract_or_reaggregate(tmp_path, monkeypatch):
    from automated_picbreeder import experiment_viewer, viewer_cache

    def unexpected(*args, **kwargs):
        pytest.fail('Save pipeline reread reports or reaggregated them for the viewer')

    monkeypatch.setattr(experiment_viewer, 'load_report', unexpected)
    monkeypatch.setattr(experiment_viewer, 'batch_aggregate', unexpected)
    run_batch(strategy_factory=RandomSelectionStrategy, output_dir=tmp_path / 'batch', runs=2,
              settings=ExperimentSettings(steps=2, size=4), progress=None)


def test_invalid_jobs_and_cli_options(tmp_path):
    from automated_picbreeder.experiment_viewer import main
    for jobs in [0, -1, True, 1.5]:
        with pytest.raises(ValueError, match='jobs'):
            write_viewer(tmp_path, jobs=jobs)
    main([str(tmp_path), '--no-open', '--refresh', '--jobs', '1'])
    assert (tmp_path / 'index.html').exists()
