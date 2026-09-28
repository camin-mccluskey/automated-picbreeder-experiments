"""Selection-time value targets for actual offspring; no candidate probing."""

from copy import deepcopy
from numbers import Integral, Real

import numpy as np

from .evaluation import Evaluation, validate_image
from .selection_strategies import _image_hash


def _finite_vector(values, *, name, count=None):
    array = np.asarray(values)
    if (array.ndim != 1 or array.dtype.kind not in 'fiu' or
            not np.isfinite(array).all() or (count is not None and len(array) != count)):
        raise ValueError(f'{name} must be a finite numeric vector' + (f' of length {count}.' if count else '.'))
    return array.astype(np.float64, copy=True)


def reference_ranks(values, reference):
    """Compare each value against nine fixed anchors, preserving average-tie ranks."""
    values = _finite_vector(values, name='Values')
    reference = _finite_vector(reference, name='Reference', count=9)
    below = (reference[None, :] < values[:, None]).sum(axis=1)
    equal = (reference[None, :] == values[:, None]).sum(axis=1)
    return np.clip((below + (equal - 1) / 2) / 8, 0, 1)


class FrozenOffspringValue:
    """Own one selection-time value function, including an independent observer.

    The caller supplies measurements from the original nine-image grid, before
    training on it, and the mean of the preceding grid. evaluate_offspring accepts
    the subsequent retained-parent-plus-eight-children grid. Pixel matching guards
    chronology but cannot establish genome ancestry; audit saved genome links too.
    """

    def __init__(self, *, reference_mean, novelty_reference, comprehension_reference,
                 observer, selected_parent, origin_generation, comprehension_weight=.5):
        if isinstance(origin_generation, bool) or not isinstance(origin_generation, Integral) or origin_generation < 1:
            raise ValueError('Origin generation must be >= 1; the initial grid has no informed value.')
        if (isinstance(comprehension_weight, bool) or not isinstance(comprehension_weight, Real)
                or not np.isfinite(comprehension_weight) or not 0 <= comprehension_weight <= 1):
            raise ValueError('comprehension_weight must be finite and in [0, 1].')
        validate_image(selected_parent)
        if selected_parent.ndim != 3 or selected_parent.shape[-1] != 3:
            raise ValueError('Expected an RGB parent image.')
        mean = np.asarray(reference_mean)
        if (mean.shape != selected_parent.shape or mean.dtype.kind not in 'fiu'
                or not np.isfinite(mean).all() or np.any((mean < 0) | (mean > 1))):
            raise ValueError('Reference mean must match the RGB image shape and contain finite pixels in [0, 1].')
        self._mean = mean.astype(np.float64, copy=True)
        self._novelty = _finite_vector(novelty_reference, name='Novelty reference', count=9)
        self._comprehension = _finite_vector(comprehension_reference, name='Comprehension reference', count=9)
        for reference in (self._novelty, self._comprehension):
            if np.any((reference < 0) | (reference > 1)):
                raise ValueError('Reference measurements must be in [0, 1].')
            reference.flags.writeable = False
        self._mean.flags.writeable = False
        self._weight = float(comprehension_weight)
        self._observer = observer.snapshot()
        self._metadata = {
            'target': 'mean_fixed_reference_offspring_value', 'version': 1,
            'origin_generation': int(origin_generation), 'reference_generation': int(origin_generation) - 1,
            'parent_image_hash': _image_hash(selected_parent),
            'reference_mean_hash': _image_hash(self._mean),
            'observer_state_hash': self._observer.state_hash(),
            'observer': deepcopy(self._observer.describe()),
            'novelty_reference': self._novelty.tolist(),
            'comprehension_reference': self._comprehension.tolist(),
            'comprehension_weight': self._weight,
            'rank_rule': 'clip((count_below + (count_equal - 1)/2)/8, 0, 1); nine fixed anchors',
        }

    def describe(self):
        return deepcopy(self._metadata)

    def score(self, images):
        """Score images without changing the references, observer or any history."""
        if not images:
            raise ValueError('Cannot score an empty image list.')
        for image in images:
            validate_image(image)
            if image.shape != self._mean.shape:
                raise ValueError('Images must match the reference RGB shape.')
        pixels = np.stack(images).astype(np.float64) / 255
        novelty = np.mean((pixels - self._mean) ** 2, axis=(1, 2, 3))
        errors = _finite_vector(self._observer.predict(images)[0], name='Observer errors', count=len(images))
        if np.any((errors < 0) | (errors > 1)):
            raise ValueError('Observer errors must be in [0, 1].')
        comprehension = 1 - errors
        novelty_rank = reference_ranks(novelty, self._novelty)
        comprehension_rank = reference_ranks(comprehension, self._comprehension)
        values = (1 - self._weight) * novelty_rank + self._weight * comprehension_rank
        return Evaluation(np.column_stack((novelty, comprehension, novelty_rank, comprehension_rank, values)),
                          ('pixel_novelty', 'comprehension', 'novelty_rank', 'comprehension_rank', 'fixed_reference_value'),
                          self.describe() | {'image_hashes': [_image_hash(image) for image in images]})

    def evaluate_offspring(self, next_grid):
        """Return eight child rows and their mean, excluding the retained parent."""
        if len(next_grid) != 9:
            raise ValueError('Expected one retained parent followed by eight offspring.')
        validate_image(next_grid[0])
        if _image_hash(next_grid[0]) != self._metadata['parent_image_hash']:
            raise ValueError('Retained parent does not match the selected parent.')
        result = self.score(next_grid[1:])
        return Evaluation(result.values, result.names, result.metadata | {
            'child_count': 8, 'mean_offspring_value': float(result.values[:, -1].mean()),
            'masked_inputs_scored': 8 * 16,
        })
