"""Run-local prediction of the configured eight-child offspring value aggregate; loaded only on demand."""
import hashlib
from importlib.metadata import version

import numpy as np
import torch
from torch import nn


class _ValueNetwork(nn.Module):
    def __init__(self, context_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(64 + context_size, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, pixels, context):
        return self.head(torch.cat((self.features(pixels), context), dim=1)).flatten()


class OffspringValuePredictor:
    """Estimate the expected configured aggregate for an actual eight-child brood.

    Each fresh run learns one aggregation from completed parent transitions. The
    aggregation changes its labels, not its network or 21/23-scalar context.
    """

    def __init__(self, *, learning_rate=.001, device='cpu', value_source='predictability',
                 offspring_aggregation='max'):
        if offspring_aggregation not in ('max', 'mean'):
            raise ValueError('offspring_aggregation must be max or mean.')
        self.offspring_aggregation = offspring_aggregation
        if value_source not in ('predictability', 'imagenet'):
            raise ValueError('Predictor value_source must be predictability or imagenet.')
        if device not in ('cpu', 'mps'):
            raise ValueError('Predictor device must be cpu or mps.')
        if device == 'mps' and not torch.backends.mps.is_available():
            raise ValueError('MPS unavailable; run with Apple GPU access or use cpu.')
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('Predictor learning rate must be positive and finite.')
        self.device, self.learning_rate = device, float(learning_rate)
        self.value_source = value_source
        self.context_size = 23 if value_source == 'predictability' else 21
        self.model = self.optimizer = None

    def initialize(self, seed):
        if self.model is not None:
            raise RuntimeError('Use a fresh predictor for each run.')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = _ValueNetwork(self.context_size)
        self.model = model.to(self.device).eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def describe(self):
        quality = 'comprehension' if self.value_source == 'predictability' else 'imagenet_confidence'
        context = ['raw_novelty', quality, 'current_value', 'sorted_novelty_9', f'sorted_{quality}_9']
        if self.value_source == 'predictability':
            context += ['log1p_observer_updates', 'log1p_observer_replay_size']
        return {
            'predictor': 'offspring_value', 'version': 2, 'offspring_aggregation': self.offspring_aggregation, 'device': self.device,
            'architecture': f'Conv3x3 stride2 pad1: 6->16->32->64 with ReLU; global mean; concatenate {self.context_size} scalars; {64+self.context_size}->64 ReLU->1 sigmoid',
            'inputs': 'full-resolution parent RGB/255 and selection-time novelty reference RGB image, six channels; float32',
            'context': context,
            'initialization': 'scratch, CPU, run-derived seed',
            'optimizer': {'name': 'Adam', 'lr': self.learning_rate, 'betas': [.9,.999], 'eps': 1e-8, 'weight_decay': 0},
            'loss': f'MSE against realised {self.offspring_aggregation} value of eight actual children',
            'sampling': 'uniform transitions with replacement; temporary CPU generator per update call',
            'torch_version': version('torch'), 'torch_threads': torch.get_num_threads(),
        }

    def prepare(self, images, reference_image, contexts):
        reference_pixels = np.asarray(reference_image, dtype=np.float32).transpose(2, 0, 1)
        contexts = np.asarray(contexts, dtype=np.float32)
        if contexts.shape != (len(images), self.context_size) or not np.isfinite(contexts).all():
            raise ValueError(f'Expected {self.context_size} finite context values per parent.')
        result = []
        for image, context in zip(images, contexts, strict=True):
            pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)/255
            result.append((torch.from_numpy(np.concatenate((pixels, reference_pixels)).copy()),
                           torch.from_numpy(context.copy())))
        return result

    def state_hash(self):
        digest = hashlib.sha256()
        for key, value in self.model.state_dict().items():
            digest.update(f'{key}:{value.dtype}:{tuple(value.shape)}'.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _batch(self, inputs):
        return tuple(torch.stack([item[i] for item in inputs]).to(self.device) for i in (0,1))

    def predict(self, inputs):
        self.model.eval()
        with torch.inference_mode():
            return self.model(*self._batch(inputs)).cpu().numpy().astype(np.float64)

    def train(self, replay, *, steps, batch_size, seed):
        if not replay or steps < 1 or batch_size < 1:
            raise ValueError('Training requires transitions and a positive budget.')
        generator = torch.Generator(device='cpu').manual_seed(seed)
        digest, losses = hashlib.sha256(), []
        self.model.train()
        try:
            for _ in range(steps):
                indices = torch.randint(len(replay), (batch_size,), generator=generator)
                digest.update(indices.numpy().tobytes())
                examples = [replay[i] for i in indices.tolist()]
                inputs = self._batch([e['inputs'] for e in examples])
                targets = torch.tensor([e['target'] for e in examples], dtype=torch.float32, device=self.device)
                self.optimizer.zero_grad(set_to_none=True)
                loss = (self.model(*inputs) - targets).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite offspring prediction loss.')
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach()))
        finally:
            self.model.eval()
        return {'updates': steps, 'sampled_examples': steps*batch_size,
                'losses': losses, 'mean_loss': float(np.mean(losses)), 'sample_sha256': digest.hexdigest()}
