"""Selection-time target semantics; no offspring forecasts are fitted here."""

from copy import deepcopy

import numpy as np
import pytest

from automated_picbreeder.offspring_value import FrozenImageNetOffspringValue, FrozenOffspringValue, reference_ranks
from automated_picbreeder.selection_strategies import _percentile_ranks


def picture(value):
    return np.full((8, 8, 3), value, dtype=np.uint8)


class ControlledObserver:
    def __init__(self):
        self.error = .25

    def snapshot(self):
        return deepcopy(self)

    def predict(self, images):
        return np.full(len(images), self.error), None

    def state_hash(self):
        return str(self.error)

    def describe(self):
        return {'observer': 'controlled'}


def context(**kwargs):
    options = dict(reference_image=np.zeros((8, 8, 3)),
                   novelty_reference=np.linspace(0, 1, 9),
                   comprehension_reference=np.linspace(0, 1, 9),
                   observer=ControlledObserver(), selected_parent=picture(100),
                   origin_generation=1, comprehension_weight=0)
    return FrozenOffspringValue(**(options | kwargs))


def test_reference_ranks_match_existing_ranks_with_ties_and_permutations():
    rng = np.random.default_rng(7)
    for _ in range(50):
        values = rng.integers(0, 5, 9) / 4
        np.testing.assert_array_equal(reference_ranks(values, values), _percentile_ranks(values))
        np.testing.assert_array_equal(reference_ranks(values, rng.permutation(values)), _percentile_ranks(values))


def test_fixed_reference_bounds_ties_and_between_reference_values():
    np.testing.assert_array_equal(reference_ranks([0, .5, 1], np.full(9, .5)), [0, .5, 1])
    assert reference_ranks([.0625], np.linspace(0, 1, 9))[0] == .0625


@pytest.mark.parametrize('values,reference', [([np.nan], np.zeros(9)),
    ([1], np.full(9, np.inf)), ([1], np.zeros(8)), ([[1]], np.zeros(9)),
    ([1], np.zeros((3, 3))), (['x'], np.zeros(9))])
def test_invalid_ranks_are_rejected(values, reference):
    with pytest.raises(ValueError):
        reference_ranks(values, reference)


def test_fixed_reference_distinguishes_broods_that_self_ranking_cannot():
    frozen = context()
    low = frozen.evaluate_offspring([picture(100)] + [picture(0)] * 8)
    high = frozen.evaluate_offspring([picture(100)] + [picture(255)] * 8)
    assert low.metadata['mean_offspring_value'] == 0
    assert high.metadata['mean_offspring_value'] == 1
    assert _percentile_ranks(np.zeros(8)).mean() == _percentile_ranks(np.ones(8)).mean() == .5


def test_parent_excluded_and_identical_children_are_not_deduplicated():
    frozen = context(offspring_aggregation='mean')
    result = frozen.evaluate_offspring([picture(100)] + [picture(0)] * 7 + [picture(255)])
    assert len(result.values) == 8
    assert result.metadata['mean_offspring_value'] == .125
    assert len(result.metadata['image_hashes']) == 8


@pytest.mark.parametrize('aggregation', ['mean', 'max'])
def test_context_owns_independent_reference_arrays_observer_and_records(aggregation):
    observer = ControlledObserver()
    mean, novelty, comprehension = np.zeros((8, 8, 3)), np.linspace(0, 1, 9), np.linspace(0, 1, 9)
    frozen = context(reference_image=mean, novelty_reference=novelty,
                     comprehension_reference=comprehension, observer=observer, comprehension_weight=.5,
                     offspring_aggregation=aggregation)
    grid = [picture(100)] + [picture(128)] * 8
    before = frozen.evaluate_offspring(grid)
    observer.error = .9
    mean[:] = 1
    novelty[:] = 0
    comprehension[:] = 1
    after = frozen.evaluate_offspring(grid)
    np.testing.assert_array_equal(before.values, after.values)
    assert before.metadata == after.metadata
    after.metadata['novelty_reference'][0] = 99
    assert frozen.describe()['novelty_reference'][0] == 0


@pytest.mark.parametrize('grid', [[picture(100)] * 8, [picture(100)] * 10,
    [picture(99)] + [picture(0)] * 8, [picture(100)] + [np.zeros((9, 8, 3), dtype=np.uint8)] * 8])
def test_invalid_feedback_is_rejected(grid):
    with pytest.raises(ValueError):
        context().evaluate_offspring(grid)


@pytest.mark.parametrize('options', [{'origin_generation': 0}, {'comprehension_weight': np.nan},
    {'reference_image': np.full((8, 8, 3), np.nan)}, {'novelty_reference': np.zeros(8)},
    {'comprehension_reference': np.full(9, 2.)}, {'selected_parent': np.zeros((8, 8), dtype=np.uint8)}])
def test_invalid_context_is_rejected(options):
    with pytest.raises(ValueError):
        context(**options)


def test_nonfinite_observer_errors_cannot_become_training_labels():
    observer = ControlledObserver()
    observer.error = np.nan
    with pytest.raises(ValueError, match='Observer'):
        context(observer=observer).evaluate_offspring([picture(100)] * 9)


@pytest.mark.parametrize('kind', ['predictability', 'imagenet'])
@pytest.mark.parametrize('aggregation,expected', [('mean', .125), ('max', 1.)])
def test_configured_target_aggregates_all_eight_actual_children(kind, aggregation, expected):
    options = dict(reference_image=np.zeros((8, 8, 3)),
                   novelty_reference=np.linspace(0, 1, 9),
                   selected_parent=picture(255), origin_generation=1,
                   comprehension_weight=0, offspring_aggregation=aggregation)
    if kind == 'predictability':
        frozen = FrozenOffspringValue(**options, observer=ControlledObserver(),
                                     comprehension_reference=np.linspace(0, 1, 9))
        measurements = {}
    else:
        frozen = FrozenImageNetOffspringValue(**options, evaluator_metadata={},
                                             confidence_reference=np.linspace(0, 1, 9))
        measurements = {'confidence': np.full(8, .5)}
    # Repeated children count individually; the exceptionally good parent is excluded.
    result = frozen.evaluate_offspring([picture(255)] + [picture(0)] * 7 + [picture(255)],
                                      **measurements)
    assert result.metadata['offspring_value'] == expected
    assert result.metadata['mean_offspring_value'] == .125
    assert result.metadata['max_offspring_value'] == 1.
    assert result.metadata['offspring_aggregation'] == aggregation
    assert frozen.describe()['offspring_aggregation'] == aggregation
    assert 'mean' not in frozen.describe()['target']
    assert len(result.values) == len(result.metadata['image_hashes']) == 8
    low = frozen.evaluate_offspring([picture(255)] + [picture(0)] * 8, **measurements)
    assert low.metadata['offspring_value'] == 0


def test_max_is_default_and_reference_metadata_is_neutral():
    frozen = context()
    result = frozen.evaluate_offspring([picture(100)] + [picture(0)] * 7 + [picture(255)])
    assert result.metadata['offspring_value'] == 1.
    assert frozen.describe()['offspring_aggregation'] == 'max'
    assert 'reference_image_hash' in frozen.describe()
    assert 'reference_mean_hash' not in frozen.describe()


@pytest.mark.parametrize('aggregation', ['median', '', None, 1, True])
def test_invalid_aggregation_rejected_before_observer_snapshot(aggregation):
    class UnavailableObserver:
        def snapshot(self):
            raise AssertionError('Must reject configuration before constructing snapshot')
    with pytest.raises(ValueError, match='offspring_aggregation'):
        context(offspring_aggregation=aggregation, observer=UnavailableObserver())
