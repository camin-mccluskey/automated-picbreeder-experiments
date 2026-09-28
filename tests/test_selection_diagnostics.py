import numpy as np
import pytest

from automated_picbreeder.selection_diagnostics import synthetic_images, mean_fill_errors


def test_synthetic_families_are_reproducible_and_heldout_images_are_distinct():
    train, again, heldout = synthetic_images(101), synthetic_images(101), synthetic_images(202)
    assert set(train) == {'flat', 'gradient', 'pattern', 'noise'}
    for family in train:
        np.testing.assert_array_equal(train[family], again[family])
        assert np.asarray(train[family]).shape == (6, 32, 32, 3)
    seen = {image.tobytes() for images in train.values() for image in images}
    assert all(image.tobytes() not in seen for images in heldout.values() for image in images)


def test_mean_fill_baseline_predicts_flat_colours_but_not_independent_noise():
    pytest.importorskip('torch')
    families = synthetic_images(7, per_family=1)
    assert mean_fill_errors(families['flat'])[0] < 1e-12
    assert mean_fill_errors(families['noise'])[0] > .05
