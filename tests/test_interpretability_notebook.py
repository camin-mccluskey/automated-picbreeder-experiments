import json
import random
import base64
from io import BytesIO
import re

import numpy as np
from PIL import Image

from automated_picbreeder.cppn import genome_to_dict, load_config, new_genome
from automated_picbreeder.interpretability import SavedNetwork, replay_experiment
from automated_picbreeder.interpretability_notebook import CPPNInspector, activation_pixels


def test_widget_interventions_reset_sweep_and_saved_replay(tmp_path):
    config = load_config()
    genome = new_genome(0, config, random.Random(7))
    original = genome_to_dict(genome)
    inspector = CPPNInspector(SavedNetwork(genome, config, 6, {}), save_dir=tmp_path)
    assert "<svg" in inspector.graph.value
    assert "data:image/png;base64" in inspector.maps.value
    inspector.target.value = (-3, 2)
    inspector.value.value += .4
    assert inspector.current.intervention.kind == "weight"
    assert "Distance from centre (-3) → Brightness output (2)" in inspector.selection.value
    source, destination = inspector.target.value
    rendered_maps = [np.asarray(Image.open(BytesIO(base64.b64decode(data))))
                     for data in re.findall(r'src="data:image/png;base64,([^"]+)"', inspector.maps.value)]
    assert len(rendered_maps) == 3
    for actual, trace, key in zip(rendered_maps, [inspector.baseline, inspector.baseline, inspector.current],
                                   [source, destination, destination]):
        np.testing.assert_array_equal(actual, activation_pixels(trace.activations[key], inspector.limits[key]))
    np.testing.assert_array_equal(inspector.current.activations[source], inspector.baseline.activations[source])
    inspector.target.value = (-1, 0)
    assert "Horizontal position (-1) → Hue output (0)" in inspector.selection.value
    assert inspector.current.intervention is None
    assert "Brightness output" not in inspector.maps.value
    inspector.reset_button.click()
    assert inspector.current.intervention is None
    np.testing.assert_array_equal(inspector.current.rgb, inspector.baseline.rgb)
    for kind in ["bias", "clamp", "disable"]:
        inspector.kind.value = kind
        inspector.apply_button.click()
        assert inspector.current.intervention.kind == kind
        assert "Could not" not in inspector.status.value
    inspector.kind.value = "weight"
    inspector.count.value = 3
    inspector.sweep_button.click()
    assert len(inspector.sweep_results) == 3
    inspector.observation.value = "Exploratory sweep."
    inspector.save_button.click()
    assert "Saved 3 interventions" in inspector.status.value
    directory = next(tmp_path.iterdir())
    assert json.loads((directory / "analysis.json").read_text())["observation"] == "Exploratory sweep."
    _, results = replay_experiment(directory)
    for actual, expected in zip(results, inspector.sweep_results):
        np.testing.assert_array_equal(actual.rgb, expected.rgb)
    inspector.reset_button.click()
    assert inspector.strip.value == ""
    assert not inspector.sweep_results
    assert genome_to_dict(genome) == original


def test_fixed_scales_empty_edge_controls_and_clamp_output(tmp_path):
    config = load_config()
    genome = new_genome(0, config, random.Random(7))
    for connection in genome.connections.values():
        connection.enabled = False
    inspector = CPPNInspector(SavedNetwork(genome, config, 4, {}), save_dir=tmp_path)
    assert inspector.apply_button.disabled
    assert inspector.sweep_button.disabled
    assert "Not evaluated" in inspector.graph.value
    inspector.kind.value = "clamp"
    inspector.target.value = 2
    limits = inspector.limits.copy()
    inspector.value.value = .5
    np.testing.assert_array_equal(inspector.current.hsb[..., 2], .5)
    assert inspector.limits == limits
    inspector.low.value = 2
    inspector.high.value = 1
    inspector.sweep_button.click()
    assert "Could not sweep" in inspector.status.value
    np.testing.assert_array_equal(activation_pixels(np.array([[-1, 0, 1, 2]]), 1),
                                  [[[0, 0, 255], [255, 255, 255], [255, 0, 0], [255, 0, 0]]])
