"""Inspect saved CPPNs without running breeding, selection or a classifier."""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import neat
import numpy as np
from PIL import Image

from .cppn import (OUTPUT_MAPPING, coordinates, genome_from_dict, genome_to_dict,
                   hsb_to_rgb, load_config, png_bytes, render)


@dataclass
class SavedNetwork:
    genome: object
    config: object
    size: int
    source: dict

    def trace(self, intervention=None, *, size=None):
        return _trace_neat_network(self, intervention, size=size)

    def export_definition(self, directory):
        self.config.save(directory / "cppn.cfg")
        return {"genome": genome_to_dict(self.genome),
                "config": (directory / "cppn.cfg").read_text()}


@dataclass(frozen=True)
class Intervention:
    """Absolute value, not a delta. Clamp replaces a node's post-activation value.

    kind: weight, bias, disable, or clamp. Disable requires value=None.
    Only one intervention is applied at a time, always to the original network.
    """

    kind: str
    target: int | tuple[int, int]
    value: float | None = None


@dataclass
class Trace:
    genome: object
    activations: dict
    hsb: np.ndarray
    rgb: np.ndarray
    intervention: Intervention | None
    # Connections actually consumed by the native feed-forward evaluator.
    active_edges: set


def _config_from_text(text):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cppn.cfg"
        path.write_text(text)
        return load_config(path)


def load_network(path, genome_id=None):
    """Load a current version-2 session and verify its saved image exactly.

    Accept a run directory or session.json. Uses the embedded configuration,
    including for checkpoints whose image paths are relative to the run root.
    """
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "session.json"
    payload = path.read_bytes()
    data = json.loads(payload)
    if data.get("format_version") != 2 or data.get("output_mapping") != OUTPUT_MAPPING:
        raise ValueError("Expected a version-2 session using picbreeder_hsb_v1.")
    key = data["selected"] if genome_id is None else genome_id
    if key is None:
        raise ValueError("No saved selection; supply an explicit genome_id.")
    record = next((g for g in data["genomes"] if g["key"] == key), None)
    if record is None:
        raise ValueError(f"Genome {key} is not in this session.")
    config = _config_from_text(data["config"])
    genome = genome_from_dict(record)
    root = path.parent.parent if path.parent.name == "checkpoints" else path.parent
    image_path = root / record["image"]
    if sha256(image_path.read_bytes()).hexdigest() != record["image_sha256"]:
        raise ValueError("Saved image hash does not match the session record.")
    with Image.open(image_path) as image:
        matches = np.array_equal(render(genome, config, data["size"]), np.asarray(image))
    if not matches:
        raise ValueError("Loaded network does not reproduce the saved image in this environment.")
    return SavedNetwork(genome, config, data["size"], {
        "session": str(path), "session_sha256": sha256(payload).hexdigest(), "genome_id": key,
    })


def trace_network(network, intervention=None, *, size=None):
    """Trace a saved native network or an imported reference network."""
    return network.trace(intervention, size=size)


def _trace_neat_network(network, intervention=None, *, size=None):
    """Trace NEAT's native evaluator; never change the supplied genome.

    Only evaluated nodes and inputs have maps. Unused genes are retained in
    Trace.genome, but no activation is invented for an unevaluated node.
    """
    size = network.size if size is None else size
    points = coordinates(size).reshape(-1, 3)
    if network.config.output_mapping != OUTPUT_MAPPING:
        raise ValueError("Unsupported output_mapping.")
    genome = deepcopy(network.genome)
    if intervention is not None:
        kind, target, value = intervention.kind, intervention.target, intervention.value
        if kind not in ("weight", "bias", "disable", "clamp"):
            raise ValueError(f"Unknown intervention kind: {kind}")
        if kind == "disable":
            if value is not None:
                raise ValueError("Disable requires value=None.")
        elif value is None or not np.isfinite(value):
            raise ValueError("Intervention value must be finite.")
        if kind in ("weight", "disable"):
            if target not in genome.connections:
                raise ValueError(f"Unknown connection: {target}")
            connection = genome.connections[target]
            if not connection.enabled:
                raise ValueError("Connection is already disabled.")
            if kind == "weight":
                connection.weight = float(value)
            else:
                connection.enabled = False
        else:
            if target not in genome.nodes:
                raise ValueError(f"Unknown non-input node: {target}")
            if kind == "bias":
                genome.nodes[target].bias = float(value)
    native = neat.nn.FeedForwardNetwork.create(genome, network.config)
    evaluated = {row[0] for row in native.node_evals}
    if intervention is not None and intervention.kind == "clamp":
        if intervention.target not in evaluated:
            raise ValueError("Cannot clamp a node outside the active computation.")
        # Keep NEAT's aggregation, ordering and propagation. Replace only the
        # chosen activation function, so descendants consume the clamped value.
        native.node_evals = [
            (node, (lambda z: float(intervention.value)) if node == intervention.target else act,
             agg, bias, response, links)
            for node, act, agg, bias, response, links in native.node_evals
        ]
    keys = [*native.input_nodes, *sorted(evaluated)]
    values = {key: np.empty(len(points)) for key in keys}
    outputs = np.empty((len(points), 3))
    for index, point in enumerate(points):
        outputs[index] = native.activate(point)
        for key in keys:
            values[key][index] = native.values[key]
    if not np.isfinite(outputs).all() or any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError("Non-finite activation; reduce the intervention magnitude.")
    hsb = outputs.reshape(size, size, 3)
    return Trace(genome, {k: v.reshape(size, size) for k, v in values.items()},
                 hsb, hsb_to_rgb(hsb), intervention,
                 {(source, node) for node, *_, links in native.node_evals for source, _ in links})


def sweep(network, intervention, values, *, size=None):
    """Independent absolute-value interventions; no cumulative edits."""
    if intervention.kind == "disable":
        raise ValueError("Disable is a single intervention, not a numeric sweep.")
    return [trace_network(network, replace(intervention, value=float(v)), size=size) for v in values]


def save_experiment(directory, network, interventions, *, observation="", size=None):
    """Save a self-contained baseline, exact interventions and resulting PNGs.

    The directory must be new. Source sessions are never modified. Replay uses
    the recorded baseline and configuration even if the source run is moved.
    """
    size = network.size if size is None else size
    interventions = list(interventions)
    baseline = trace_network(network, size=size)
    results = [trace_network(network, item, size=size) for item in interventions]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    definition = network.export_definition(directory)
    images = {}
    for name, result in [("baseline.png", baseline),
                         *[(f"intervention-{i:03d}.png", r) for i, r in enumerate(results)]]:
        payload = png_bytes(result.rgb)
        (directory / name).write_bytes(payload)
        images[name] = sha256(payload).hexdigest()
    record = {
        "format_version": 1, "source": network.source, "size": size,
        **definition,
        "interventions": [asdict(item) for item in interventions],
        "observation": observation, "images_sha256": images,
        "versions": {p: version(p) for p in ("automated-picbreeder", "neat-python", "numpy", "pillow")},
    }
    (directory / "analysis.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return directory


def replay_experiment(directory):
    """Recompute a saved experiment and reject results that differ from its PNGs."""
    directory = Path(directory)
    record = json.loads((directory / "analysis.json").read_text())
    if record["format_version"] != 1:
        raise ValueError("Unsupported analysis format.")
    if "reference_model" in record:
        from .picbreeder_reference import ReferenceNetwork
        network = ReferenceNetwork(record["reference_model"], size=record["size"])
    else:
        network = SavedNetwork(genome_from_dict(record["genome"]),
                               _config_from_text(record["config"]), record["size"], record["source"])
    specs = [Intervention(item["kind"], tuple(item["target"]) if isinstance(item["target"], list)
                          else item["target"], item["value"]) for item in record["interventions"]]
    baseline = trace_network(network)
    results = [trace_network(network, item) for item in specs]
    for name, result in zip(["baseline.png", *[f"intervention-{i:03d}.png" for i in range(len(specs))]],
                            [baseline, *results]):
        payload = (directory / name).read_bytes()
        if sha256(payload).hexdigest() != record["images_sha256"][name]:
            raise ValueError(f"Saved image hash mismatch: {name}")
        with Image.open(directory / name) as image:
            if not np.array_equal(result.rgb, np.asarray(image)):
                raise ValueError(f"Replay differs from saved image: {name}")
    return baseline, results
