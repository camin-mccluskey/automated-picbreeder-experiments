"""Human decisions share diagnostics without erasing branches or changing breeding."""

import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest

from automated_picbreeder.experiment import ExperimentSettings, run_experiment
from automated_picbreeder.experiment_metrics import build_run_metrics, save_run_metrics
from automated_picbreeder.notebook import CPPNPlayground
from automated_picbreeder.selection_strategies import RandomSelectionStrategy


def read(path):
    return json.loads(path.read_text())


def test_human_and_automatic_metrics_match_for_the_same_choices(tmp_path):
    auto = tmp_path / "auto"
    run_experiment(selection_strategy=RandomSelectionStrategy(), output_dir=auto,
                   settings=ExperimentSettings(seed=9, steps=4, size=4), progress=None)
    data = read(auto / "session.json")
    ui = CPPNPlayground(seed=9, size=4, save_dir=tmp_path)
    for i, event in enumerate(e for e in data["events"] if e["action"] == "select"):
        if i:
            ui.evolve()
        ui.select(event["position"])
    before = random.getstate(), ui.rng.getstate()
    human = ui.save()
    assert before == (random.getstate(), ui.rng.getstate())
    a, h = read(auto / "metrics.json"), read(human / "metrics.json")
    assert h["status"] == "complete"
    assert h["summary"]["candidate_presentations"] == a["summary"]["candidate_presentations"] == 36
    assert h["summary"]["decisions"] == 4
    assert h["summary"]["run_seconds"] is None
    assert h["summary"]["interaction_seconds"] >= 0
    assert h["settings"]["seed"] == 9
    assert h["settings"]["size"] == 4
    assert h["final_ancestry"] == a["final_ancestry"]
    for ar, hr in zip(a["generations"], h["generations"], strict=True):
        assert hr["selection_mode"] == "human"
        assert (human / hr["grid"]).is_file()
        assert ar["selected_id"] == hr["selected_id"]
        for key in ("pixel_mse_previous", "pixel_mse_nearest_earlier", "parent_retained", "image_unchanged",
                    "unchanged_image_streak", "children_identical_to_parent_fraction", "distinct_child_images",
                    "display_novelty_selected", "display_novelty_grid_mean"):
            assert ar["metrics"][key] == hr["metrics"][key]
    assert read(human / "session.json")["genomes"] == data["genomes"]
    assert build_run_metrics(human) == h


def test_branches_reselections_resets_and_current_ancestry_are_not_flattened(tmp_path):
    ui = CPPNPlayground(seed=7, size=4, save_dir=tmp_path)
    ui.select(2)
    ui.select(3)  # A changed choice on the same initial grid, not nine new candidates.
    root = ui.selected_id
    ui.strength.value = .45
    ui.topology.value = False
    ui.evolve()
    ui.select(1)
    abandoned = ui.selected_id
    ui.back()
    assert ui.selected_id == root
    ui.strength.value = .1
    ui.evolve()
    ui.select(2)
    child = ui.selected_id
    ui.evolve()
    ui.select(1)
    grandchild = ui.selected_id
    ui.back()  # Final image is restored child, not the most recent selection.
    assert ui.selected_id == child
    output = ui.save()
    report = read(output / "metrics.json")
    rows = report["generations"]
    assert report["final_selected_id"] == child
    assert report["final_image"] == f"images/{child:08d}.png"
    assert [node["genome_id"] for node in report["final_ancestry"]] == [root, child]
    assert rows[-1]["selected_id"] == grandchild
    assert not rows[-1]["on_final_ancestry"]
    assert not next(r for r in rows if r["selected_id"] == abandoned)["on_final_ancestry"]
    assert [r["display_index"] for r in rows] == [0, 0, 1, 3, 4]
    assert rows[1]["metrics"]["parent_retained"] is None  # Root reselection is not retention.
    assert rows[2]["mutation"] == {"strength": .45, "topology": False}
    assert rows[3]["mutation"] == {"strength": .1, "topology": False}
    assert rows[3]["preceding_actions"] == ["back", "evolve"]
    assert report["summary"]["displayed_grids"] == 6
    assert report["summary"]["candidate_presentations"] == 54
    assert report["summary"]["generated_genomes"] == 33
    assert report["summary"]["backtracks"] == 2
    assert report["summary"]["selection_revisions"] == 1
    assert len(report["display_history"]) == 6
    ui.reset()  # Save with no current selection, preserving all earlier history.
    report = read(ui.save() / "metrics.json")
    assert report["final_image"] is None
    assert report["final_ancestry"] == []
    assert report["summary"]["resets"] == 1
    assert report["summary"]["candidate_presentations"] == 63


def test_empty_selection_history_and_implicit_parent_retention(tmp_path):
    ui = CPPNPlayground(seed=7, size=4, save_dir=tmp_path)
    report = read(ui.save() / "metrics.json")
    assert report["generations"] == []
    assert report["summary"]["candidate_presentations"] == 9
    assert report["final_image"] is None
    ui.select(4)
    root = ui.selected_id
    ui.evolve()
    ui.evolve()  # No new click: the UI still has the retained parent selected.
    report = read(ui.save() / "metrics.json")
    assert report["summary"]["decisions"] == 1
    assert report["summary"]["candidate_presentations"] == 27
    assert len(report["final_ancestry"]) == 1
    assert report["final_selected_id"] == root


def test_offline_classifier_is_separate_reusable_and_never_relabels_human_decisions(tmp_path):
    from automated_picbreeder.posthoc_evaluation import evaluate_session_imagenet
    from automated_picbreeder.evaluation import Evaluation

    class Classifier:
        def __init__(self): self.count = 0
        def describe(self): return {"evaluator": "controlled", "checkpoint": "fixed"}
        def evaluate(self, images):
            self.count += len(images)
            scores = np.array([.5 + im.mean()/510 for im in images])
            return Evaluation(np.column_stack((scores, 1-scores)), ("a", "b"), self.describe())

    ui = CPPNPlayground(seed=7, size=4, save_dir=tmp_path)
    ui.select(0)
    ui.evolve()
    ui.select(2)
    output = ui.save()
    original = (output / "session.json").read_bytes()
    evaluator = Classifier()
    evaluation = evaluate_session_imagenet(output, evaluator=evaluator)
    assert evaluator.count == 17  # Once per saved genome, not nine per decision.
    assert evaluation["evaluated_images"] == 17
    report = save_run_metrics(output)
    assert evaluator.count == 17
    assert (output / "session.json").read_bytes() == original
    assert report["summary"]["classifier_images"] == 0
    assert report["summary"]["posthoc_classifier_images"] == 17
    assert report["posthoc_evaluation"]["evaluator"]["checkpoint"] == "fixed"
    for row in report["generations"]:
        assert row["selection_mode"] == "human"
        assert row["metrics"]["selected_score"] is None
        assert "imagenet_selected_confidence" not in row["metrics"]
        assert .5 <= row["metrics"]["posthoc_imagenet_selected_confidence"] <= 1
        assert all("posthoc_top_class" in c for c in row["candidates"])
    assert build_run_metrics(output) == report
    # The same optional analysis works on automated runs with unchanged choices.
    auto = tmp_path / "random"
    run_experiment(selection_strategy=RandomSelectionStrategy(), output_dir=auto,
                   settings=ExperimentSettings(steps=2, size=4), progress=None)
    evaluate_session_imagenet(auto, evaluator=Classifier())
    assert build_run_metrics(auto)["summary"]["posthoc_classifier_images"] == 17


def test_posthoc_snapshot_mismatch_is_rejected(tmp_path):
    from automated_picbreeder.posthoc_evaluation import load_posthoc_evaluation

    ui = CPPNPlayground(size=4, save_dir=tmp_path)
    output = ui.save()
    (output / "posthoc_imagenet.json").write_text(json.dumps({"session_sha256": "wrong snapshot"}))
    with pytest.raises(ValueError, match="different session snapshot"):
        load_posthoc_evaluation(output)


def test_human_save_needs_no_torch_and_preserves_all_grid_image_references(tmp_path):
    code = '''
import builtins, json, sys
from pathlib import Path
original_import = builtins.__import__
def without_torch(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision'}:
        raise ImportError('Torch deliberately unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_torch
from automated_picbreeder.notebook import CPPNPlayground
p = CPPNPlayground(size=4, save_dir=Path(sys.argv[1]))
p.select_buttons[1].click()
p.evolve_button.click()
p.select_buttons[2].click()
p.back_button.click()
p.save_button.click()
assert not p.errors.outputs
assert p.last_saved_dir is not None
report = json.loads((p.last_saved_dir / 'metrics.json').read_text())
for row in report['generations']:
    assert (p.last_saved_dir / row['grid']).is_file()
    assert (p.last_saved_dir / row['selected_image']).is_file()
    assert all((p.last_saved_dir / c['image']).is_file() for c in row['candidates'])
assert 'torch' not in sys.modules
'''
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True, capture_output=True, text=True)


def test_notebook_keeps_exploration_settings_and_metrics_examples_are_opt_in():
    notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks/01_cppn_selection.ipynb").read_text())
    cells = ["".join(c["source"]) for c in notebook["cells"]]
    assert any('CPPNPlayground(seed=9, size=96, save_dir=ROOT / "runs")' in c for c in cells)
    assert any("evaluate_session_imagenet" in c for c in cells)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and "evaluate_session_imagenet" in "".join(cell["source"]):
            assert all(line.startswith("#") or not line.strip() for line in cell["source"])
