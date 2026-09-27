"""Selection rules operate on an evaluated grid, independently of breeding."""

from typing import Protocol

import numpy as np

from .evaluation import Evaluation


class Selector(Protocol):
    def select(self, evaluation: Evaluation) -> int:
        """Return a zero-based candidate position, without modifying the grid."""
        ...

    def describe(self) -> dict:
        """Return JSON-compatible identification/configuration for run records."""
        ...


class MaximumClassConfidence:
    """Choose the candidate with the highest score for any class.

    The first candidate wins exact ties. The shared breeding loop puts the
    retained parent first, so a tied offspring cannot displace it.
    """

    def select(self, evaluation: Evaluation) -> int:
        if len(evaluation.values) == 0:
            raise ValueError("Cannot select from an empty candidate grid.")
        return int(np.argmax(evaluation.values.max(axis=1)))

    def describe(self) -> dict:
        return {"selector": "maximum_class_confidence", "rule": "argmax_i max_class scores[i, class]",
                "ties": "first candidate; retained parent is first after initialization"}
