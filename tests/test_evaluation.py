import numpy as np
import pytest

from automated_picbreeder.evaluation import Evaluation, ImageEvaluator, validate_image


def test_generic_evaluator_can_return_non_probability_measurements():
    class Measurements:
        def evaluate(self, images):
            return Evaluation(np.array([[image.mean(), -image.std()] for image in images]),
                              ("brightness", "negative contrast"))

    evaluator: ImageEvaluator = Measurements()  # No inheritance or classifier dependency.
    images = [np.array([[0, 255]], dtype=np.uint8), np.full((2, 2), 20, dtype=np.uint8)]
    result = evaluator.evaluate(images)
    np.testing.assert_allclose(result.values, [[127.5, -127.5], [20, 0]])
    assert result.names == ("brightness", "negative contrast")


@pytest.mark.parametrize("values,names", [
    (np.array([0.1, 0.2]), ("a", "b")),
    (np.ones((2, 2)), ("a",)),
    (np.array([[np.nan]]), ("a",)),
    (np.array([[np.inf]]), ("a",)),
    (np.array([[1j]]), ("a",)),
    (np.array([["x"]]), ("a",)),
    (np.zeros((1, 0)), ()),
])
def test_invalid_evaluation_fails_early(values, names):
    with pytest.raises(ValueError):
        Evaluation(values, names)


def test_empty_batch_and_repeated_display_names_are_valid():
    result = Evaluation(np.empty((0, 2)), ("same name", "same name"))
    assert result.values.shape == (0, 2)


@pytest.mark.parametrize("image", [
    np.zeros((2, 2), dtype=float), np.zeros((3, 2, 2), dtype=np.uint8),
    np.zeros((2, 2, 4), dtype=np.uint8), np.zeros((0, 2), dtype=np.uint8),
])
def test_ambiguous_image_inputs_are_rejected(image):
    with pytest.raises(ValueError):
        validate_image(image)
