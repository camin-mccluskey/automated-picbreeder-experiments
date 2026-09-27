from copy import deepcopy
from dataclasses import replace
import json
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from automated_picbreeder.cppn import hsb_to_rgb
from automated_picbreeder.interpretability import replay_experiment, save_experiment, sweep, trace_network
from automated_picbreeder.interpretability_notebook import CPPNInspector
from automated_picbreeder.picbreeder_reference import REFERENCE_DIR, load_skull, reference_coordinates


def dense_forward(size, parameter_id=None, delta=0):
    """Independent execution of the authors' published float32 layer matrices."""
    with np.load(REFERENCE_DIR / "layered_weights.npz", allow_pickle=False) as data:
        matrices = {k: data[k].copy() for k in data.files}
    if parameter_id is not None:
        offset = 0
        for key in sorted(matrices):
            matrix = matrices[key]
            if offset <= parameter_id < offset + matrix.size:
                matrix.flat[parameter_id - offset] += np.float32(delta)
                break
            offset += matrix.size
    values = reference_coordinates(size)
    for layer in range(13):
        values = values @ matrices[f"Dense_{layer}"]
        if layer < 12:
            values = np.concatenate([values[..., :15], 2*np.exp(-values[..., 15:19]**2)-1,
                                     values[..., 19:21], np.sin(values[..., 21:22])], axis=-1)
    return values


def test_original_genome_and_published_parameter_mapping():
    network = load_skull()
    with zipfile.ZipFile(REFERENCE_DIR / "original.zip") as archive:
        root = ET.fromstring(archive.read(archive.namelist()[0])).find("genome")
    assert len(root.find("nodes")) == 23
    original = {(l.find("source").get("branch") + "_" + l.find("source").get("id"),
                 l.find("target").get("branch") + "_" + l.find("target").get("id")):
                float(l.findtext("weight")) for l in root.find("links")}
    with np.load(REFERENCE_DIR / "layered_weights.npz", allow_pickle=False) as data:
        flat = np.concatenate([data[k].ravel() for k in sorted(data.files)])
    for name, spec in network.presets.items():
        mapping = network.record["presets"][name]
        source_weight = original[mapping["source"], mapping["target"]]
        assert np.float32(source_weight) == spec.value == flat[mapping["parameter_id"]]


@pytest.mark.parametrize("name", ["Mouth opening", "Eye winking", "Eye width", "Jaw width"])
def test_sparse_reference_sweeps_match_published_layered_model(name):
    network = load_skull(size=48)
    spec = network.presets[name]
    original = deepcopy(network.record)
    traces = sweep(network, spec, [spec.value + d for d in [-1, -.5, 0, .5, 1]])
    for delta, trace in zip([-1, -.5, 0, .5, 1], traces):
        expected = dense_forward(48, network.record["presets"][name]["parameter_id"], delta)
        np.testing.assert_allclose(trace.hsb, expected, atol=3e-5, rtol=0)
        assert np.abs(trace.rgb.astype(int)-hsb_to_rgb(expected).astype(int)).max() <= 1
    assert np.mean(traces[0].rgb != traces[-1].rgb) > .1
    assert network.record == original
    assert network.genome.connections[spec.target].weight == spec.value


def test_baseline_matches_dense_and_records_png_discrepancy():
    result = trace_network(load_skull(size=256))
    np.testing.assert_allclose(result.hsb, dense_forward(256), atol=3e-5, rtol=0)
    # The repository PNG is not pixel-identical to its published parameters.
    # This is a visual-reference check, not a claim of exact PNG reproduction.
    published = np.asarray(Image.open(REFERENCE_DIR / "published.png").convert("RGB"))
    difference = np.abs(result.rgb.astype(int)-published.astype(int))
    assert difference.mean() < 1
    assert difference.max() <= 12


def test_reference_widget_presets_and_self_contained_replay(tmp_path):
    network = load_skull(size=12)
    inspector = CPPNInspector(network, save_dir=tmp_path)
    spec = network.presets["Mouth opening"]
    inspector.target.value = spec.target
    inspector.value.value = spec.value + .5
    assert inspector.current.intervention == replace(spec, value=spec.value+.5)
    assert "bias · input" in inspector.graph.value
    assert "Source:" in inspector.maps.value
    inspector.save_button.click()
    directory = next(tmp_path.iterdir())
    _, results = replay_experiment(directory)
    np.testing.assert_array_equal(results[0].rgb, inspector.current.rgb)
    assert json.loads((directory / "analysis.json").read_text())["reference_model"]["source"]["model"].startswith("evolved")
    for kind in ("clamp", "disable", "bias"):
        target = spec.target if kind == "disable" else spec.target[1]
        intervention = replace(spec, kind=kind, target=target, value=None if kind == "disable" else .2)
        path = save_experiment(tmp_path / kind, network, [intervention])
        replay_experiment(path)


def test_reference_rejects_bad_intervention():
    network = load_skull(size=4)
    spec = network.presets["Eye winking"]
    for bad in [replace(spec, value=float("nan")), replace(spec, target=(-90, 8)),
                replace(spec, kind="clamp", target=-4), replace(spec, kind="unknown")]:
        with pytest.raises(ValueError):
            trace_network(network, bad)
