"""The original evolved Picbreeder skull, using the FER authors' forward rules.

This is an inspection-only reference, independent of our NEAT mutation config.
Model data and attribution are in reference_data/skull. No pickle is loaded.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .cppn import coordinates, hsb_to_rgb
from .interpretability import Intervention, Trace


REFERENCE_DIR = Path(__file__).with_name("reference_data") / "skull"


def reference_activation(name, value):
    """Unscaled, signed activation functions from the authors' cppn.py."""
    if name in ("identity", "cache"):
        return value
    if name == "gaussian":
        return 2 * np.exp(-value**2) - 1
    if name == "sin":
        return np.sin(value)
    if name == "cos":
        return np.cos(value)
    if name == "tanh":
        return np.tanh(value)
    if name == "sigmoid":
        return np.tanh(value / 2)  # stable equivalent of 2*sigmoid(value)-1
    raise ValueError(f"Unsupported reference activation: {name}")


def reference_coordinates(size):
    coordinates(size)  # shared size validation
    # FER's meshgrid(indexing='ij') with inputs='y,x,d,b' is equivalent
    # to ordinary xy indexing with inputs ordered x, y, d, bias.
    x, y = np.meshgrid(np.linspace(-1, 1, size, dtype=np.float32),
                       np.linspace(-1, 1, size, dtype=np.float32))
    return np.stack((x, y, np.sqrt(x*x + y*y) * np.float32(1.4), np.ones_like(x)), axis=-1)


class ReferenceNetwork:
    """Expose a reference graph to the same tracing, sweeping and widget API."""

    def __init__(self, record, *, size=128):
        if record["format"] != "fer_picbreeder_graph_v1":
            raise ValueError("Unsupported reference graph format.")
        self.record = deepcopy(record)
        self.size = size
        self.source = deepcopy(record["source"])
        labels = {n["label"]: n["id"] for n in record["nodes"] if n["label"]}
        self.ids = {labels[label]: key for label, key in
                    [("x", -1), ("y", -2), ("d", -3), ("bias", -4),
                     ("hue", 0), ("saturation", 1), ("brightness", 2)]}
        for node in record["nodes"]:
            if node["id"] not in self.ids:
                self.ids[node["id"]] = 3 + len(self.ids) - 7
        nodes = {self.ids[n["id"]]: SimpleNamespace(activation=n["activation"], bias=0.)
                 for n in record["nodes"] if self.ids[n["id"]] >= 0}
        connections = {(self.ids[l["source"]], self.ids[l["target"]]):
                       SimpleNamespace(weight=float(np.float32(l["weight"])), enabled=True)
                       for l in record["links"]}
        self.genome = SimpleNamespace(key=record["name"], nodes=nodes, connections=connections)
        self.config = SimpleNamespace(
            genome_config=SimpleNamespace(input_keys=[-1, -2, -3, -4], output_keys=[0, 1, 2]),
            node_labels={-1: "Horizontal position", -2: "Vertical position", -3: "Scaled radius",
                         -4: "Constant bias input", 0: "Hue output", 1: "Saturation output",
                         2: "Brightness / ink output"},
        )
        self.presets = {name: Intervention("weight", (self.ids[p["source"]], self.ids[p["target"]]),
                                          p["weight"]) for name, p in record["presets"].items()}

    def export_definition(self, directory):
        return {"reference_model": deepcopy(self.record)}

    def trace(self, intervention=None, *, size=None):
        size = self.size if size is None else size
        points = reference_coordinates(size)
        genome = deepcopy(self.genome)
        clamp = None
        if intervention is not None:
            kind, target, value = intervention.kind, intervention.target, intervention.value
            if kind not in ("weight", "disable", "bias", "clamp"):
                raise ValueError(f"Unknown intervention: {kind}")
            if kind == "disable":
                if value is not None:
                    raise ValueError("Disable requires value=None.")
            elif value is None or not np.isfinite(value):
                raise ValueError("Intervention value must be finite.")
            if kind in ("weight", "disable"):
                if target not in genome.connections:
                    raise ValueError(f"Unknown connection: {target}")
                if kind == "weight":
                    genome.connections[target].weight = float(value)
                else:
                    genome.connections[target].enabled = False
            elif target not in genome.nodes:
                raise ValueError(f"Unknown non-input node: {target}")
            elif kind == "bias":
                genome.nodes[target].bias = float(value)
            else:
                clamp = (target, float(value))
        values = {key: points[..., i] for i, key in enumerate(self.config.genome_config.input_keys)}
        incoming = {key: [(a, c.weight) for (a, b), c in genome.connections.items()
                          if b == key and c.enabled] for key in genome.nodes}
        visiting = set()
        used_edges = set()

        def evaluate(key):
            if key in values:
                return values[key]
            if key in visiting:
                raise ValueError("Reference graph contains a cycle.")
            visiting.add(key)
            summed = np.full((size, size), genome.nodes[key].bias, dtype=np.float32)
            for source, weight in incoming[key]:
                summed += np.float32(weight) * evaluate(source)
                used_edges.add((source, key))
            value = reference_activation(genome.nodes[key].activation, summed)
            if clamp is not None and key == clamp[0]:
                value = np.full_like(value, clamp[1])
            values[key] = value
            visiting.remove(key)
            return value

        with np.errstate(over="ignore", invalid="ignore"):
            hsb = np.stack([evaluate(k) for k in self.config.genome_config.output_keys], axis=-1)
        if any(not np.isfinite(v).all() for v in values.values()):
            raise ValueError("Non-finite activation; reduce the intervention magnitude.")
        if clamp is not None and clamp[0] not in values:
            raise ValueError("Cannot clamp a node outside the active computation.")
        return Trace(genome, values, hsb, hsb_to_rgb(hsb), intervention, used_edges)


def load_skull(*, size=128):
    """Load the bundled, attributed evolved skull without network or ML dependencies."""
    return ReferenceNetwork(json.loads((REFERENCE_DIR / "model.json").read_text()), size=size)
