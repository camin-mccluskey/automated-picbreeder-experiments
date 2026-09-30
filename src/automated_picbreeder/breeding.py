"""The breeding loop shared by the human interface and automated experiments."""

import random
from copy import deepcopy

from .cppn import load_config, mutate_genome, new_genome


class BreedingSession:
    """Nine roots, then one unchanged selected parent plus eight mutations."""

    def __init__(self, seed=7):
        self.seed = seed
        self.config = load_config()
        self.rng = random.Random(seed)
        self.genomes = {}
        self.parents = {}
        self.events = []
        self.history = []
        self.selected_id = None
        self._next_id = 0
        BreedingSession.reset(self)

    def _register(self, genome, parent=None):
        self.genomes[genome.key] = genome
        self.parents[genome.key] = parent
        self._next_id += 1
        return genome.key

    def _remember(self):
        self.history.append((self.candidates.copy(), self.selected_id, self.round))

    def reset(self):
        if hasattr(self, "candidates"):
            self._remember()
        self.config.genome_config.innovation_tracker.reset_generation()
        self.candidates = [self._register(new_genome(self._next_id, self.config, self.rng)) for _ in range(9)]
        self.selected_id = None
        self.round = 0
        self.events.append({"action": "reset", "displayed": self.candidates.copy()})

    def select(self, position, *, decision=None):
        """Record a human choice, or a selection strategy's complete decision."""
        if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < len(self.candidates):
            raise ValueError("Selection position is outside the displayed grid.")
        if decision is not None:
            decision.validate(len(self.candidates))
            if decision.position != position:
                raise ValueError("Decision position does not match the selected position.")
        evaluation = None if decision is None else decision.evaluation
        measurements = None if evaluation is None else {
            "values": evaluation.values.tolist(), "names": list(evaluation.names),
            "metadata": deepcopy(evaluation.metadata),
        }
        selection = None if decision is None else {
            "mode": decision.mode,
            "scores": None if decision.scores is None else decision.scores.tolist(),
        }
        if decision is not None and decision.metadata:
            selection["metadata"] = deepcopy(decision.metadata)
        self.selected_id = self.candidates[position]
        self.events.append({"action": "select", "position": position, "genome": self.selected_id,
                            "displayed": self.candidates.copy(), "evaluation": measurements,
                            "decision": selection})

    @property
    def selected_genome(self):
        if self.selected_id is None:
            raise ValueError("Select an image in the grid first.")
        return self.genomes[self.selected_id]

    def evolve(self, strength=0.2, topology=True):
        parent = self.selected_genome
        self._remember()
        self.config.genome_config.innovation_tracker.reset_generation()
        children = []
        for _ in range(8):
            child = mutate_genome(parent, self._next_id, self.config, self.rng,
                                  strength=strength, topology=topology)
            children.append(self._register(child, parent.key))
        before = self.candidates.copy()
        self.candidates = [parent.key, *children]
        self.round += 1
        self.events.append({"action": "evolve", "parent": parent.key, "previous_display": before,
                            "displayed": self.candidates.copy(), "strength": strength,
                            "topology": topology})

    def back(self):
        if not self.history:
            return
        self.candidates, self.selected_id, self.round = self.history.pop()
        self.events.append({"action": "back", "displayed": self.candidates.copy(),
                            "selected": self.selected_id})
