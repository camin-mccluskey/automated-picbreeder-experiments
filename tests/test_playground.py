import json
import hashlib
import random
from colorsys import hsv_to_rgb
from io import BytesIO

import neat
import numpy as np
import pytest
from PIL import Image

from automated_picbreeder.cppn import (
    CONFIG_PATH, coordinates, genome_from_dict, genome_to_dict, load_config,
    mutate_genome, new_genome, png_bytes, render,
)
from automated_picbreeder.notebook import CPPNPlayground


def test_output_mapping_survives_config_export_and_rejects_unknown_mapping(tmp_path):
    config = load_config()
    genome = new_genome(0, config, random.Random(7))
    exported = tmp_path / "cppn.cfg"
    config.save(exported)
    restored = load_config(exported)
    assert restored.output_mapping == "picbreeder_hsb_v1"
    np.testing.assert_array_equal(render(genome, config, 8), render(genome, restored, 8))
    exported.write_text(CONFIG_PATH.read_text().replace("picbreeder_hsb_v1", "unknown"))
    with pytest.raises(ValueError, match="Unsupported output_mapping"):
        load_config(exported)
    config.output_mapping = "unknown"
    with pytest.raises(ValueError, match="Unsupported output_mapping"):
        render(genome, config, 8)


def test_render_matches_pixel_evaluation_and_json_roundtrip():
    config = load_config()
    genome = new_genome(0, config, random.Random(7))
    pixels = render(genome, config, 12)
    assert pixels.shape == (12, 12, 3)
    assert pixels.dtype == np.uint8
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    for y, x in ((0, 0), (3, 9), (11, 11)):
        h, s, b = net.activate(coordinates(12)[y, x])
        rgb = hsv_to_rgb((h + 1) % 1, min(1, max(0, s)), min(1, abs(b)))
        np.testing.assert_array_equal(pixels[y, x], [int(255 * c + .5) for c in rgb])
    restored = genome_from_dict(json.loads(json.dumps(genome_to_dict(genome))))
    np.testing.assert_array_equal(pixels, render(restored, config, 12))
    decoded = Image.open(BytesIO(png_bytes(pixels)))
    assert decoded.mode == "RGB"
    np.testing.assert_array_equal(pixels, np.asarray(decoded))


@pytest.mark.parametrize("hsb,rgb", [
    ((0, 1, 1), (255, 0, 0)), ((1/3, 1, 1), (0, 255, 0)),
    ((2/3, 1, 1), (0, 0, 255)), ((1, 1, 1), (255, 0, 0)),
    ((.7, 0, .5), (128, 128, 128)), ((.3, 1, 0), (0, 0, 0)),
    ((-.25, 1, 1), (128, 0, 255)), ((1.5, 2, -2), (0, 255, 255)),
    ((.1, -.5, -.8), (204, 204, 204)), ((0, 0, 2.5/255), (3, 3, 3)),
])
def test_hsb_channel_order_and_limits(monkeypatch, hsb, rgb):
    class ConstantNetwork:
        def activate(self, point):
            return hsb
    monkeypatch.setattr(neat.nn.FeedForwardNetwork, "create", lambda *args: ConstantNetwork())
    image = render(None, load_config(), 2)
    np.testing.assert_array_equal(image, np.tile(rgb, (2, 2, 1)))


def test_output_activations_are_not_overridden_at_initialization():
    config = load_config()
    config.genome_config.activation_default = "identity"
    genome = new_genome(0, config, random.Random(7))
    assert all(genome.nodes[k].activation == "identity" for k in config.genome_config.output_keys)


def test_mutation_preserves_parent_and_can_freeze_structure():
    config = load_config()
    rng = random.Random(3)
    parent = new_genome(0, config, rng)
    before = genome_to_dict(parent)
    clone = mutate_genome(parent, 1, config, rng, strength=0, topology=False)
    np.testing.assert_array_equal(render(parent, config, 12), render(clone, config, 12))
    changed = mutate_genome(parent, 2, config, rng, strength=0.8, topology=False)
    assert genome_to_dict(parent) == before
    assert set(changed.nodes) == set(parent.nodes)
    assert set(changed.connections) == set(parent.connections)
    assert all(changed.nodes[k].activation == n.activation for k, n in parent.nodes.items())
    assert all(changed.connections[k].enabled == c.enabled for k, c in parent.connections.items())
    assert any(changed.connections[k].weight != c.weight for k, c in parent.connections.items())
    config.genome_config.node_add_prob = 1.0
    config.genome_config.node_delete_prob = 0.0
    config.genome_config.conn_add_prob = 0.0
    config.genome_config.conn_delete_prob = 0.0
    config.genome_config.activation_mutate_rate = 1.0
    config.genome_config.activation_options = ["identity"]
    grown = mutate_genome(parent, 3, config, rng, topology=True)
    assert len(grown.nodes) == len(parent.nodes) + 1
    assert all(grown.nodes[k].activation == "identity" for k in config.genome_config.output_keys)


def test_rng_isolation_and_reproducibility():
    state = random.getstate()
    a = CPPNPlayground(seed=7, size=8)
    b = CPPNPlayground(seed=7, size=8)
    for playground in (a, b):
        playground.select(2)
        playground.evolve()
        playground.select(5)
        playground.evolve()
    assert random.getstate() == state
    assert [genome_to_dict(g) for g in a.genomes.values()] == [genome_to_dict(g) for g in b.genomes.values()]


def test_widget_callbacks_branch_reset_and_export(tmp_path):
    p = CPPNPlayground(size=8, save_dir=tmp_path)
    roots = p.candidates.copy()
    assert all("2 hidden" in label.value for label in p.labels)
    assert p.evolve_button.disabled
    with pytest.raises(ValueError, match="Select"):
        p.evolve()
    p.select_buttons[3].click()
    assert p.selected_id == roots[3]
    assert not p.evolve_button.disabled
    parent_image = p.images[p.selected_id]
    p.evolve_button.click()
    children = p.candidates[1:]
    assert len(p.genomes) == 17
    assert p.candidates[0] == roots[3]
    assert p.images[p.candidates[0]] == parent_image
    assert all(p.parents[k] == roots[3] for k in children)
    p.back_button.click()
    assert p.candidates == roots
    p.evolve_button.click()
    assert set(p.candidates[1:]).isdisjoint(children)
    directory = p.save()
    data = json.loads((directory / "session.json").read_text())
    assert len(data["genomes"]) == 25
    assert (directory / "selected.png").read_bytes() == parent_image
    config = load_config(directory / "cppn.cfg")
    for record in data["genomes"]:
        np.testing.assert_array_equal(render(genome_from_dict(record), config, 8),
                                      render(p.genomes[record["key"]], p.config, 8))
    p.reset_button.click()
    assert p.selected_id is None and p.evolve_button.disabled
    assert len(p.genomes) == 34
    assert not p.errors.outputs
    # Saving after a reset retains discarded branches and has no selected PNG.
    directory = p.save()
    data = json.loads((directory / "session.json").read_text())
    assert data["format_version"] == 2
    assert len(data["genomes"]) == len(list((directory / "images").glob("*.png"))) == 34
    assert data["selected"] is None
    assert not (directory / "selected.png").exists()
    for record in data["genomes"]:
        image = (directory / record["image"]).read_bytes()
        assert image == p.images[record["key"]]
        assert record["image_sha256"] == hashlib.sha256(image).hexdigest()
    assert all(e["evaluation"] is None for e in data["events"] if e["action"] == "select")
