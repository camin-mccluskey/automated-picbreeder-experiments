"""Small, inspectable CPPNs using NEAT-Python's genomes and evaluator."""

from contextlib import contextmanager
from colorsys import hsv_to_rgb
from configparser import ConfigParser
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import random

import neat
from neat.innovation import InnovationTracker
import numpy as np
from PIL import Image


CONFIG_PATH = Path(__file__).with_name("cppn.cfg")
OUTPUT_MAPPING = "picbreeder_hsb_v1"


class CPPNConfig(neat.Config):
    """Preserve the rendering setting alongside NEAT's own configuration."""

    def save(self, filename):
        super().save(filename)
        with open(filename, "a") as handle:
            handle.write(f"\n[CPPNRendering]\noutput_mapping = {self.output_mapping}\n")


def load_config(path=CONFIG_PATH):
    config = CPPNConfig(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, str(path),
    )
    if config.genome_config.num_outputs != 3:
        raise ValueError("Expected three CPPN outputs ordered hue, saturation, brightness.")
    parser = ConfigParser()
    parser.read(path)
    config.output_mapping = parser.get("CPPNRendering", "output_mapping")
    if config.output_mapping != OUTPUT_MAPPING:
        raise ValueError(f"Unsupported output_mapping: {config.output_mapping!r}. Expected {OUTPUT_MAPPING!r}.")
    # Normally supplied by NEAT's population/reproduction objects.
    config.genome_config.innovation_tracker = InnovationTracker()
    return config


@contextmanager
def use_rng(rng):
    """Isolate NEAT's module-level RNG in this single-threaded notebook."""
    previous = random.getstate()
    random.setstate(rng.getstate())
    try:
        yield
    finally:
        rng.setstate(random.getstate())
        random.setstate(previous)


def new_genome(key, config, rng):
    with use_rng(rng):
        genome = neat.DefaultGenome(key)
        genome.configure_new(config.genome_config)
    return genome


def mutate_genome(parent, key, config, rng, strength=0.2, topology=True):
    """Copy a parent and mutate it. Never alter the selected parent itself."""
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("Mutation strength must be finite and non-negative.")
    # Shallow copy: preserve the shared innovation tracker and node ID counter.
    # Deepcopying NEAT 2.0's config can itself advance its node ID counter.
    from copy import copy

    settings = copy(config.genome_config)
    settings.weight_mutate_power = strength
    settings.bias_mutate_power = strength
    if not topology:
        for name in ("conn_add_prob", "conn_delete_prob", "node_add_prob",
                     "node_delete_prob", "enabled_mutate_rate", "activation_mutate_rate"):
            setattr(settings, name, 0.0)
    child = deepcopy(parent)
    child.key = key
    child.fitness = None
    with use_rng(rng):
        child.mutate(settings)
    config.genome_config.node_indexer = settings.node_indexer
    return child


def coordinates(size=96):
    """Pixel inputs: x left-to-right, y top-to-bottom, Euclidean radius."""
    if not isinstance(size, int) or size < 2:
        raise ValueError("Image size must be an integer of at least 2.")
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    return np.stack((x, y, np.hypot(x, y)), axis=-1)


def render(genome, config, size=96):
    """Map mutable HSB outputs to RGB using Picbreeder-VLM's convention.

    Hue wraps, saturation clips, and brightness takes its absolute value then
    clips. HSB is HSV, not HSL. There is no per-image normalization.
    """
    if config.output_mapping != OUTPUT_MAPPING:
        raise ValueError(f"Unsupported output_mapping: {config.output_mapping!r}.")
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    rgb = []
    for point in coordinates(size).reshape(-1, 3):
        h, s, b = network.activate(point)
        rgb.append(hsv_to_rgb((h + 1.0) % 1.0, min(1.0, max(0.0, s)), min(1.0, abs(b))))
    # Match Picbreeder's half-up byte rounding, rather than np.rint's ties-to-even.
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8).reshape(size, size, 3)


def png_bytes(pixels):
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def genome_to_dict(genome):
    def gene_values(gene):
        values = {attr.name: getattr(gene, attr.name) for attr in gene._gene_attributes}
        values["key"] = gene.key
        if hasattr(gene, "innovation"):
            values["innovation"] = gene.innovation
        return values

    # Preserve insertion order: it determines mutation RNG draw order.
    return {"key": genome.key, "nodes": [gene_values(n) for n in genome.nodes.values()],
            "connections": [gene_values(c) for c in genome.connections.values()]}


def genome_from_dict(data):
    """Restore a genome for inspection/rendering (not an evolution checkpoint)."""
    genome = neat.DefaultGenome(data["key"])
    for values in data["nodes"]:
        node = neat.genes.DefaultNodeGene(values["key"])
        for attr in node._gene_attributes:
            setattr(node, attr.name, values[attr.name])
        genome.nodes[node.key] = node
    for values in data["connections"]:
        key = tuple(values["key"])
        conn = neat.genes.DefaultConnectionGene(key, innovation=values["innovation"])
        conn.weight, conn.enabled = values["weight"], values["enabled"]
        genome.connections[key] = conn
    return genome
