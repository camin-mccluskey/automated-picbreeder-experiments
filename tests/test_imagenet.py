"""Offline checks; no checkpoint download in the default test suite."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from automated_picbreeder.imagenet import ImageNetEvaluator


@pytest.fixture
def evaluator(monkeypatch, tmp_path):
    """Use controlled logits, but real checkpoint metadata and preprocessing."""
    class TinyClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.seen = []

        def forward(self, batch):
            self.seen.append((len(batch), torch.is_grad_enabled(), self.training))
            logits = torch.zeros((len(batch), 1000), device=batch.device)
            logits[:, 10] = batch.mean(dim=(1, 2, 3)) * self.weight
            return logits

    model = TinyClassifier()

    def load_weights(url, model_dir, **kwargs):
        assert kwargs["check_hash"] and kwargs["weights_only"]
        (Path(model_dir) / url.rsplit("/", 1)[1]).write_bytes(b"test checkpoint")
        return model.state_dict()

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", load_weights)
    monkeypatch.setattr(torchvision.models, "get_model", lambda *a, **k: model)
    return ImageNetEvaluator(batch_size=2, cache_dir=tmp_path)


def test_preprocessing_uses_checkpoint_transform_and_repeats_grayscale(evaluator):
    from PIL import Image
    image = np.arange(96 * 96, dtype=np.uint8).reshape(96, 96)
    expected = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()(Image.fromarray(image).convert("RGB"))
    actual = evaluator.preprocess(image)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (3, 224, 224)
    torch.testing.assert_close(actual, evaluator.preprocess(np.repeat(image[..., None], 3, axis=2)))


def test_scores_preserve_order_duplicates_and_all_classes(evaluator):
    black, white = np.zeros((12, 12), dtype=np.uint8), np.full((12, 12), 255, dtype=np.uint8)
    result = evaluator.evaluate([black, white, black])
    assert result.values.shape == (3, 1000)
    assert len(result.names) == 1000
    assert result.names[10] == "brambling"
    # 1,000 float32 softmax terms accumulate a few units of rounding error.
    np.testing.assert_allclose(result.values.sum(axis=1), 1, atol=1e-5)
    np.testing.assert_array_equal(result.values[0], result.values[2])
    assert result.values[1, 10] > result.values[0, 10]
    assert evaluator._model.seen == [(2, False, False), (1, False, False)]
    assert all(not p.requires_grad for p in evaluator._model.parameters())
    assert evaluator.evaluate([]).values.shape == (0, 1000)
    json.dumps(result.metadata)
    assert result.metadata["weights"] == "ResNet18_Weights.IMAGENET1K_V1"
    result.metadata["preprocessing"]["mean"][0] = -1
    assert evaluator.describe()["preprocessing"]["mean"][0] == 0.485


def test_rgb_channels_survive_preprocessing(evaluator):
    red = np.zeros((96, 96, 3), dtype=np.uint8)
    red[..., 0] = 255
    actual = evaluator.preprocess(red)
    metadata = evaluator.describe()["preprocessing"]
    expected = (np.array([1, 0, 0]) - metadata["mean"]) / metadata["std"]
    np.testing.assert_allclose(actual[:, 0, 0].numpy(), expected, rtol=1e-6)
    assert evaluator.evaluate([red]).values.shape == (1, 1000)


def test_invalid_images_do_not_partially_evaluate_a_batch(evaluator):
    with pytest.raises(ValueError, match="uint8"):
        evaluator.evaluate([np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8))])
    assert evaluator._model.seen == []


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"batch_size": True}, {"weights": "DEFAULT"}])
def test_invalid_configuration_fails_before_download(kwargs):
    with pytest.raises(ValueError):
        ImageNetEvaluator(**kwargs)
