"""Image measurements, independent of CPPN genomes and selection algorithms."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

ImageArray = NDArray[np.uint8]


@dataclass
class Evaluation:
    """One row per input image; one column per named measurement.

    Values are finite real numbers, not necessarily probabilities. Column index
    is the identifier; display names may repeat (as ImageNet labels do).
    Selection policies decide how to use the columns, including which direction
    is preferable. Metadata describes the evaluator and any reference context.
    """

    values: NDArray[np.floating]
    names: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.names = tuple(self.names)
        values = np.asarray(self.values)
        if values.ndim != 2 or values.shape[1] != len(self.names) or not self.names:
            raise ValueError("Expected a 2D image-by-measurement array matching non-empty names.")
        if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError("Evaluation values must be finite real numbers.")
        if any(not isinstance(name, str) or not name for name in self.names):
            raise ValueError("Measurement names must be non-empty strings.")
        self.values = values.astype(np.float64 if values.dtype.kind in "iu" else values.dtype, copy=True)


class ImageEvaluator(Protocol):
    """Implement this method to supply measurements to a future search policy.

    Images are uint8 arrays shaped H×W (grayscale) or H×W×3 (RGB).
    Preserve input order, including duplicates. An empty input produces zero
    rows with the evaluator's usual columns. Evaluation does not select parents,
    update an archive or mutate genomes. History-dependent evaluators can be
    constructed with an explicit reference set; record its identity in metadata
    and update it separately from evaluation.
    """

    def evaluate(self, images: Sequence[ImageArray]) -> Evaluation: ...


def validate_image(image: ImageArray) -> None:
    """Reject ambiguous ranges/layouts instead of silently changing inputs."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Images must be uint8 numpy arrays with values in [0, 255].")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError("Expected H×W grayscale or H×W×3 RGB images.")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Images must have non-empty spatial dimensions.")
