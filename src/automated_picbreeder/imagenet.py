"""Frozen Torchvision ImageNet classifier implementing ImageEvaluator."""

from collections.abc import Sequence
from copy import deepcopy
import hashlib
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from .evaluation import Evaluation, ImageArray, validate_image


class ImageNetEvaluator:
    """Return all 1,000 softmax scores, without making selection decisions.

    The first construction downloads pretrained weights (ResNet-18: ~45 MB).
    Use cache_dir to keep checkpoints in a chosen directory. CPU is the explicit
    default; pass device='mps' or 'cuda' to opt into an available accelerator.
    Select a concrete weight version rather than Torchvision's moving DEFAULT.
    """

    def __init__(
        self, model_name: str = "resnet18", weights: str = "IMAGENET1K_V1",
        device: str = "cpu", batch_size: int = 16,
        cache_dir: str | Path | None = None,
    ):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if weights == "DEFAULT":
            raise ValueError("Choose an explicit checkpoint version, e.g. IMAGENET1K_V1.")
        try:
            import torch
            from torchvision.models import get_model, get_model_weights
        except ImportError as exc:
            raise ImportError("Install the ImageNet dependencies with: uv sync --extra imagenet") from exc

        self._torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use device='cpu'.")
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable; use device='cpu'.")
        weight_enum = get_model_weights(model_name)
        try:
            checkpoint = weight_enum[weights]
        except KeyError as exc:
            raise ValueError(f"Unknown weights {weights!r} for {model_name}.") from exc
        self.names = tuple(checkpoint.meta.get("categories", ()))
        if len(self.names) != 1000:
            raise ValueError("Choose a Torchvision ImageNet-1K classification model.")
        self.batch_size = batch_size
        self._transform = checkpoint.transforms()
        directory = Path(cache_dir) if cache_dir is not None else Path(torch.hub.get_dir()) / "checkpoints"
        # Download official data weights, with the published filename hash check.
        # No code is loaded through torch.hub.
        state = torch.hub.load_state_dict_from_url(
            checkpoint.url, model_dir=str(directory), check_hash=True, weights_only=True,
        )
        # Model construction temporarily initializes random weights; do not
        # consume the caller's RNG state even though we immediately replace them.
        with torch.random.fork_rng(devices=[]):
            self._model = get_model(model_name, weights=None)
        self._model.load_state_dict(state)
        self._model.requires_grad_(False)
        self._model.eval()
        self._model.to(self.device)

        checkpoint_path = directory / Path(urlparse(checkpoint.url).path).name
        with checkpoint_path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        self._metadata = {
            "evaluator": "torchvision_imagenet", "model": model_name,
            "weights": f"{weight_enum.__name__}.{checkpoint.name}",
            "checkpoint_url": checkpoint.url, "checkpoint_sha256": digest,
            "scores": "softmax over all 1000 logits; higher means greater class confidence",
            "column_identity": "zero-based ImageNet class index; names are display labels",
            "input": "uint8 HxW or HxWx3; grayscale duplicated across RGB channels",
            "preprocessing": {
                "resize_size": self._transform.resize_size,
                "crop_size": self._transform.crop_size,
                "interpolation": self._transform.interpolation.value,
                "antialias": self._transform.antialias,
                "mean": self._transform.mean, "std": self._transform.std,
                "description": repr(self._transform),
            },
            "device": str(self.device), "batch_size": batch_size,
            "versions": {p: version(p) for p in ("torch", "torchvision", "numpy", "pillow")},
        }

    def describe(self) -> dict:
        """JSON-compatible checkpoint and preprocessing provenance."""
        return deepcopy(self._metadata)

    def preprocess(self, image: ImageArray):
        """Return the actual normalized RGB input tensor, on CPU, for inspection."""
        validate_image(image)
        return self._transform(Image.fromarray(image).convert("RGB"))

    def evaluate(self, images: Sequence[ImageArray]) -> Evaluation:
        """Evaluate in bounded batches, preserving row order and every class."""
        for image in images:
            validate_image(image)
        torch = self._torch
        parts = []
        self._model.eval()
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = torch.stack([self.preprocess(image) for image in images[start:start + self.batch_size]])
                logits = self._model(batch.to(self.device))
                if not isinstance(logits, torch.Tensor) or logits.shape != (len(batch), len(self.names)):
                    raise ValueError("Classifier must return one ImageNet logit vector per image.")
                parts.append(logits.softmax(dim=1).cpu().numpy())
        values = np.concatenate(parts) if parts else np.empty((0, len(self.names)), dtype=np.float32)
        return Evaluation(values, self.names, self.describe())
