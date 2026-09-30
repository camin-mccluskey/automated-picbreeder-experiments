"""Masked-image observer. Imported only by predictability experiments."""

import hashlib
from copy import copy, deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.models import ResNet18_Weights, resnet18

from .evaluation import validate_image


IMAGE_SIZE = 96
MASK_GRID = 4
MASK_COUNT = MASK_GRID ** 2
TILE_SIZE = IMAGE_SIZE // MASK_GRID


def masked_inputs(targets, tile_ids):
    """Hide one of sixteen equal square tiles, with an explicit visibility channel."""
    if (not isinstance(targets, torch.Tensor) or targets.ndim != 4 or
            targets.shape[1:] != (3, IMAGE_SIZE, IMAGE_SIZE) or not targets.is_floating_point()):
        raise ValueError("Targets must be floating RGB tensors shaped Nx3x96x96.")
    if (not isinstance(tile_ids, torch.Tensor) or tile_ids.ndim != 1 or
            len(tile_ids) != len(targets) or
            tile_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
        raise ValueError("Supply one integer tile ID per target, in [0, 15].")
    ids = tile_ids.tolist()
    if any(tile < 0 or tile >= MASK_COUNT for tile in ids):
        raise ValueError("Tile IDs must be in [0, 15].")
    size = targets.shape[-1]
    tile_size = TILE_SIZE
    visible = targets.new_ones((len(targets), 1, size, size))
    for row, tile in enumerate(ids):
        y, x = divmod(tile, MASK_GRID)
        visible[row, :, y * tile_size:(y + 1) * tile_size, x * tile_size:(x + 1) * tile_size] = 0
    mean = targets.new_tensor([.485, .456, .406])[None, :, None, None]
    std = targets.new_tensor([.229, .224, .225])[None, :, None, None]
    return torch.cat((((targets - mean) / std) * visible, visible), dim=1), visible


def masked_mse(predictions, targets, visible):
    """One loss per example, normalized only by hidden RGB pixel count."""
    hidden = 1 - visible
    return ((predictions - targets).square() * hidden).sum((1, 2, 3)) / (hidden.sum((1, 2, 3)) * 3)


class MaskedImageObserver:
    """Identical trainable ResNet18 architecture for both initialization arms.

    initialize receives a run-owned seed; train receives a fresh update seed.
    Initialization and sampling use CPU RNGs on either device. The model is created
    lazily so describe can be saved before any initialization or weight download.
    """

    def __init__(self, *, initialization="random", learning_rate=.001, cache_dir=None, device="cpu"):
        if initialization not in ("random", "imagenet"):
            raise ValueError("initialization must be random or imagenet.")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite.")
        if device not in ("cpu", "mps"):
            raise ValueError("Observer device must be cpu or mps.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable in this process; use cpu or run with Apple GPU access.")
        self.device = device
        self.size = IMAGE_SIZE
        self.tile_size = TILE_SIZE
        self.mask_count = MASK_COUNT
        self.initialization = initialization
        self.learning_rate = float(learning_rate)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path(torch.hub.get_dir()) / "checkpoints"
        self.model = self.optimizer = None
        self.checkpoint_sha256 = None

    def initialize(self, seed):
        if self.model is not None:
            raise RuntimeError("Observer already initialized; use a fresh instance per run.")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = resnet18(weights=None)
            rgb_weights = model.conv1.weight.detach().clone()
            model.conv1 = torch.nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                model.conv1.weight[:, :3].copy_(rgb_weights)
                model.conv1.weight[:, 3].zero_()
            model.fc = torch.nn.Linear(512, 3 * self.size * self.size)
        if self.initialization == "imagenet":
            checkpoint = ResNet18_Weights.IMAGENET1K_V1
            state = torch.hub.load_state_dict_from_url(
                checkpoint.url, model_dir=str(self.cache_dir), check_hash=True, weights_only=True,
            )
            # Preserve the identically seeded new head and zero mask channel.
            backbone = {key: value for key, value in state.items() if not key.startswith("fc.")}
            backbone["conv1.weight"] = torch.cat((state["conv1.weight"], torch.zeros_like(state["conv1.weight"][:, :1])), dim=1)
            missing, unexpected = model.load_state_dict(backbone, strict=False)
            if set(missing) != {"fc.weight", "fc.bias"} or unexpected:
                raise ValueError("Checkpoint does not match the ResNet18 backbone.")
            with (self.cache_dir / checkpoint.url.rsplit("/", 1)[-1]).open("rb") as handle:
                self.checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        self.model = model.to(self.device).eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def describe(self):
        return {
            "observer": "masked_resnet18", "version": 2, "initialization": self.initialization,
            "weights": "ResNet18_Weights.IMAGENET1K_V1" if self.initialization == "imagenet" else None,
            "checkpoint_sha256": self.checkpoint_sha256, "device": self.device,
            "size": self.size, "tile_size": self.tile_size, "mask_count": self.mask_count,
            "hidden_fraction": 1 / self.mask_count,
            "architecture": f"ResNet18, four input channels (new mask weights zero), linear 512->{3*self.size*self.size}, sigmoid, RGB {self.size}x{self.size}",
            "preprocessing": f"PIL bilinear resize to {self.size}x{self.size} RGB, [0,1] targets; ImageNet normalization before masking inputs",
            "masks": f"16 row-major {self.tile_size}x{self.tile_size} tiles; one hidden per example, explicit visibility channel",
            "loss": "MSE on hidden RGB pixels only; scoring averages all 16 tiles before training",
            "optimizer": {"name": "Adam", "lr": self.learning_rate, "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0},
            "training": "all backbone and head weights; BatchNorm updates only in training mode",
            "sampling": "CPU torch.Generator seeded per update; image indices then tile indices via randint per batch",
            "versions": {name: version(name) for name in ("torch", "torchvision", "numpy", "pillow")},
            "torch_threads": torch.get_num_threads(),
        }

    def prepare(self, image):
        validate_image(image)
        resized = Image.fromarray(image).convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        pixels = np.array(resized, dtype=np.float32).transpose(2, 0, 1) / 255
        return torch.from_numpy(pixels.copy())

    def state_hash(self):
        digest = hashlib.sha256()
        for key, value in self.model.state_dict().items():
            digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def snapshot(self):
        """Copy weights and buffers for frozen inference, without RNG or optimizer state."""
        if self.model is None:
            raise RuntimeError("Observer must be initialized before taking a snapshot.")
        frozen = copy(self)
        frozen.model = deepcopy(self.model).eval().requires_grad_(False)
        frozen.optimizer = None
        return frozen

    def _forward(self, inputs):
        return self.model(inputs).sigmoid().reshape(-1, 3, self.size, self.size)

    def predict(self, images):
        """Return pre-update errors and mosaics of the sixteen hidden-tile predictions."""
        self.model.eval()
        errors, reconstructions = [], []
        with torch.inference_mode():
            for image in images:
                targets = self.prepare(image).to(self.device)[None].expand(self.mask_count, -1, -1, -1)
                inputs, visible = masked_inputs(targets, torch.arange(self.mask_count))
                prediction = self._forward(inputs)
                errors.append(float(masked_mse(prediction, targets, visible).mean()))
                reconstructions.append((prediction * (1 - visible)).sum(0).permute(1, 2, 0).cpu().numpy())
        return np.asarray(errors), np.asarray(reconstructions)

    def train(self, replay, *, steps, batch_size, seed):
        """Fixed-budget updates on prepared images; sampling never uses global RNG."""
        if self.optimizer is None:
            raise RuntimeError("Cannot train an uninitialized or inference-only observer.")
        if not replay or steps < 1 or batch_size < 2:
            raise ValueError("Training needs replay images, positive steps and batch_size >= 2 for BatchNorm.")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        sample_hash = hashlib.sha256()
        losses = []
        self.model.train()
        try:
            for _ in range(steps):
                indices = torch.randint(len(replay), (batch_size,), generator=generator)
                tiles = torch.randint(self.mask_count, (batch_size,), generator=generator)
                sample_hash.update(indices.numpy().tobytes())
                sample_hash.update(tiles.numpy().tobytes())
                targets = torch.stack([replay[i] for i in indices.tolist()]).to(self.device)
                if targets.shape[1:] != (3, self.size, self.size):
                    raise ValueError("Replay images must match the configured observer size.")
                inputs, visible = masked_inputs(targets, tiles)
                self.optimizer.zero_grad(set_to_none=True)
                loss = masked_mse(self._forward(inputs), targets, visible).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Observer training produced a non-finite loss.")
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach()))
        finally:
            self.model.eval()
        return {"updates": steps, "sampled_examples": steps * batch_size,
                "losses": losses, "mean_loss": float(np.mean(losses)), "sample_sha256": sample_hash.hexdigest()}
