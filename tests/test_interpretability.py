from copy import deepcopy
import json
import random

import neat
import numpy as np
import pytest

from automated_picbreeder.breeding import BreedingSession
from automated_picbreeder.cppn import coordinates, genome_to_dict, load_config, mutate_genome, new_genome, render
from automated_picbreeder.interpretability import (
    Intervention, SavedNetwork, load_network, replay_experiment, save_experiment, sweep, trace_network,
)
from automated_picbreeder.persistence import SessionWriter


@pytest.fixture
def simple_network():
    config = load_config()
    genome = new_genome(42, config, random.Random(1))
    for node in genome.nodes.values():
        node.activation, node.aggregation = "identity", "sum"
        node.bias, node.response = 0.0, 1.0
    for connection in genome.connections.values():
        connection.enabled = False
    hidden = min(set(genome.nodes) - {0, 1, 2})
    genome.nodes[hidden].bias = .2
    genome.nodes[2].bias = .25
    for key, weight in [((-1, hidden), .4), ((hidden, 2), .5)]:
        genome.connections[key].enabled = True
        genome.connections[key].weight = weight
    return SavedNetwork(genome, config, 7, {"genome_id": 42}), hidden


def test_native_trace_known_computation_clamp_disable_and_bias(simple_network):
    network, hidden = simple_network
    original = genome_to_dict(network.genome)
    x = coordinates(7)[..., 0]
    baseline = trace_network(network)
    np.testing.assert_allclose(baseline.activations[hidden], .2 + .4*x)
    np.testing.assert_allclose(baseline.hsb[..., 2], .35 + .2*x)
    assert set(baseline.activations) == {-1, -2, -3, 0, 1, 2, hidden}
    np.testing.assert_array_equal(baseline.rgb, render(network.genome, network.config, 7))
    clamped = trace_network(network, Intervention("clamp", hidden, 1.0))
    np.testing.assert_allclose(clamped.hsb[..., 2], .75)
    np.testing.assert_allclose(clamped.activations[hidden], 1)
    # Clamping is a runtime intervention, not a gene edit.
    assert genome_to_dict(clamped.genome) == original
    removed = trace_network(network, Intervention("disable", (hidden, 2)))
    np.testing.assert_allclose(removed.hsb[..., 2], .25)
    assert hidden not in removed.activations
    np.testing.assert_array_equal(removed.rgb, render(removed.genome, network.config, 7))
    biased = trace_network(network, Intervention("bias", hidden, .8))
    np.testing.assert_allclose(biased.hsb[..., 2], .65 + .2*x)
    assert genome_to_dict(network.genome) == original


@pytest.mark.parametrize("seed", [1, 7, 23])
def test_native_trace_parity_after_structural_mutations(seed):
    config, rng = load_config(), random.Random(seed)
    genome = new_genome(0, config, rng)
    for key in range(1, 12):
        genome = mutate_genome(genome, key, config, rng)
    network = SavedNetwork(genome, config, 9, {})
    traced = trace_network(network)
    native = neat.nn.FeedForwardNetwork.create(genome, config)
    for y, x in [(0, 0), (4, 6), (8, 8)]:
        native.activate(coordinates(9)[y, x])
        for key, values in traced.activations.items():
            assert values[y, x] == native.values[key]
    np.testing.assert_array_equal(traced.rgb, render(genome, config, 9))


def test_sweeps_are_absolute_independent_and_preserve_baseline(simple_network):
    network, hidden = simple_network
    original = genome_to_dict(network.genome)
    results = sweep(network, Intervention("weight", (-1, hidden), 0), [.8, -.2, .4, .8])
    x = coordinates(7)[..., 0]
    for result, weight in zip(results, [.8, -.2, .4, .8]):
        np.testing.assert_allclose(result.hsb[..., 2], .35 + .5*weight*x)
    np.testing.assert_array_equal(results[0].rgb, results[-1].rgb)
    np.testing.assert_array_equal(results[2].rgb, trace_network(network).rgb)
    assert genome_to_dict(network.genome) == original


@pytest.mark.parametrize("selection_strategy", [None, {"selection_strategy": "imagenet"}])
def test_load_shared_session_selection_explicit_id_and_checkpoint(tmp_path, selection_strategy):
    session = BreedingSession(7)
    session.select(2)
    session.evolve()
    writer = SessionWriter(tmp_path / "run", size=8, selection_strategy=selection_strategy)
    writer.save(session, checkpoint=1)
    loaded = load_network(writer.directory)
    assert loaded.genome.key == session.selected_id
    assert loaded.genome.key != max(session.genomes)
    other = load_network(writer.directory / "session.json", session.candidates[-1])
    assert other.genome.key == session.candidates[-1]
    checkpoint = load_network(writer.directory / "checkpoints" / "000001.json")
    np.testing.assert_array_equal(trace_network(checkpoint).rgb, trace_network(loaded).rgb)
    # Loading uses embedded config, not a possibly edited adjacent file.
    (writer.directory / "cppn.cfg").write_text("invalid")
    load_network(writer.directory)
    with pytest.raises(ValueError, match="not in this session"):
        load_network(writer.directory, -999)
    session.reset()
    writer.save(session)
    with pytest.raises(ValueError, match="No saved selection"):
        load_network(writer.directory)
    load_network(writer.directory, session.candidates[0])


def test_load_session_from_before_selection_strategy_records(tmp_path):
    session = BreedingSession(7)
    session.select(2)
    writer = SessionWriter(tmp_path / "run", size=8)
    writer.save(session, checkpoint=0)
    # Recreate the earlier version-2 record shape, without migrating it on read.
    paths = [writer.directory / "session.json", writer.directory / "checkpoints/000000.json"]
    for path in paths:
        data = json.loads(path.read_text())
        data["metadata"].pop("selection_strategy")
        data["metadata"]["selector"] = {"selector": "maximum_class_confidence"}
        for event in data["events"]:
            event.pop("decision", None)
        path.write_text(json.dumps(data))
        original = path.read_bytes()
        loaded = load_network(path)
        assert loaded.genome.key == 2
        np.testing.assert_array_equal(trace_network(loaded).rgb, render(session.genomes[2], session.config, 8))
        assert path.read_bytes() == original


def test_loader_rejects_image_drift_and_unsupported_mapping(tmp_path):
    session = BreedingSession(7)
    session.select(0)
    writer = SessionWriter(tmp_path / "run", size=8)
    writer.save(session)
    path = writer.directory / "session.json"
    data = json.loads(path.read_text())
    changed = deepcopy(data)
    changed["output_mapping"] = "old_mapping"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="version-2"):
        load_network(path)
    path.write_text(json.dumps(data))
    image = writer.directory / data["genomes"][0]["image"]
    image.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        load_network(path)


def test_recording_replays_without_source_and_does_not_overwrite(tmp_path, simple_network):
    network, hidden = simple_network
    network.source["session"] = "/no/longer/exists/session.json"
    specs = [Intervention("clamp", hidden, .6), Intervention("disable", (hidden, 2)),
             Intervention("weight", (-1, hidden), -.2), Intervention("bias", 2, .7)]
    path = save_experiment(tmp_path / "analysis", network, specs, observation="Brightness changes.")
    baseline, replayed = replay_experiment(path)
    np.testing.assert_array_equal(baseline.rgb, trace_network(network).rgb)
    for spec, result in zip(specs, replayed):
        np.testing.assert_array_equal(result.rgb, trace_network(network, spec).rgb)
    record = json.loads((path / "analysis.json").read_text())
    assert record["observation"] == "Brightness changes."
    with pytest.raises(FileExistsError):
        save_experiment(path, network, specs)
    (path / "baseline.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        replay_experiment(path)


def test_bad_interventions_are_explicit_errors(simple_network):
    network, hidden = simple_network
    unused = max(network.genome.nodes)
    for spec in [Intervention("mystery", hidden, 0), Intervention("weight", (-99, 2), 1),
                 Intervention("bias", 2, float("nan")), Intervention("bias", -1, 0),
                 Intervention("clamp", unused, 0), Intervention("disable", (-1, hidden), 0)]:
        with pytest.raises(ValueError):
            trace_network(network, spec)
