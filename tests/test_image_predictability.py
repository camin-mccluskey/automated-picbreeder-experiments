"""Masking and matched observer initialization; no network downloads."""

import hashlib
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from automated_picbreeder.image_predictability import MaskedImageObserver, masked_inputs, masked_mse


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_hidden_pixels_cannot_enter_inputs_and_only_hidden_pixels_contribute_to_loss():
    target = torch.zeros((2, 3, 32, 32))
    inputs, visible = masked_inputs(target, torch.tensor([0, 15]))
    changed = target.clone()
    changed[0, :, :8, :8] = 1
    changed[1, :, 24:, 24:] = 1
    new_inputs, _ = masked_inputs(changed, torch.tensor([0, 15]))
    assert torch.equal(inputs, new_inputs)
    assert torch.equal(inputs[:, 3:], visible)
    assert torch.all(inputs[:, :3] * (1 - visible) == 0)
    assert torch.equal(masked_mse(target, changed, visible), torch.ones(2))
    assert torch.equal(masked_mse(visible.expand(-1, 3, -1, -1), target, visible), torch.zeros(2))


def test_random_observer_is_seeded_without_consuming_global_rng_and_scores_without_learning():
    py_before, np_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    observer = MaskedImageObserver()
    observer.initialize(7)
    assert random.getstate() == py_before
    for a, b in zip(np_before, np.random.get_state()):
        np.testing.assert_equal(a, b)
    assert torch.equal(torch_before, torch.get_rng_state())
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    before = observer.state_hash()
    errors, reconstructed = observer.predict([image, image])
    assert errors.shape == (2,) and reconstructed.shape == (2, 32, 32, 3)
    assert np.isfinite(errors).all() and np.all((errors >= 0) & (errors <= 1))
    assert errors[0] == errors[1]
    assert observer.state_hash() == before
    assert not observer.model.training
    assert all(p.grad is None for p in observer.model.parameters())


def test_pretrained_backbone_load_preserves_new_head_and_zero_mask_channel(monkeypatch, tmp_path):
    from torchvision.models import resnet18
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(99)
        state = resnet18(weights=None).state_dict()
    def load(url, *, model_dir, **kwargs):
        from pathlib import Path
        checkpoint = Path(model_dir) / url.rsplit('/', 1)[-1]
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b'controlled checkpoint')
        return state
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", load)
    scratch = MaskedImageObserver()
    pretrained = MaskedImageObserver(initialization="imagenet", cache_dir=tmp_path)
    scratch.initialize(7)
    pretrained.initialize(7)
    assert torch.equal(scratch.model.fc.weight, pretrained.model.fc.weight)
    assert torch.equal(scratch.model.fc.bias, pretrained.model.fc.bias)
    assert torch.equal(pretrained.model.conv1.weight[:, :3], state['conv1.weight'])
    assert torch.count_nonzero(pretrained.model.conv1.weight[:, 3]) == 0
    assert torch.count_nonzero(scratch.model.conv1.weight[:, 3]) == 0
    assert torch.equal(pretrained.model.layer4[1].bn2.running_var, state['layer4.1.bn2.running_var'])
    assert all(p.requires_grad for p in pretrained.model.parameters())
    assert pretrained.describe()['checkpoint_sha256'] == hashlib.sha256(b'controlled checkpoint').hexdigest()


def test_learning_reduces_hidden_pixel_error_and_seeded_updates_replay():
    images = [np.zeros((32, 32, 3), dtype=np.uint8)]
    results = []
    for _ in range(2):
        observer = MaskedImageObserver()
        observer.initialize(7)
        before = observer.predict(images)[0][0]
        torch_before = torch.get_rng_state().clone()
        record = observer.train([observer.prepare(images[0])], steps=4, batch_size=2, seed=9)
        after = observer.predict(images)[0][0]
        assert after < before
        assert torch.equal(torch_before, torch.get_rng_state())
        assert record['updates'] == 4 and record['sampled_examples'] == 8
        results.append((observer.state_hash(), record))
    assert results[0] == results[1]


def test_device_validation_does_not_silently_fall_back(monkeypatch):
    with pytest.raises(ValueError, match="device"):
        MaskedImageObserver(device="other")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="MPS is unavailable"):
        MaskedImageObserver(device="mps")


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='Apple GPU access required'))])
def test_snapshot_is_inference_only_and_independent_of_live_training(device):
    observer = MaskedImageObserver(device=device)
    with pytest.raises(RuntimeError, match="initialized"):
        observer.snapshot()
    observer.initialize(7)
    rng_before = torch.get_rng_state().clone()
    frozen = observer.snapshot()
    assert torch.equal(rng_before, torch.get_rng_state())
    assert frozen.optimizer is None
    assert all(not parameter.requires_grad for parameter in frozen.model.parameters())
    assert frozen.state_hash() == observer.state_hash()
    images = [np.zeros((32, 32, 3), dtype=np.uint8)]
    predictions = frozen.predict(images)[0]
    before = frozen.state_hash()
    observer.train([observer.prepare(images[0])], steps=2, batch_size=2, seed=9)
    assert observer.state_hash() != before
    assert frozen.state_hash() == before
    np.testing.assert_array_equal(frozen.predict(images)[0], predictions)
    with pytest.raises(RuntimeError, match="inference-only"):
        frozen.train([frozen.prepare(images[0])], steps=1, batch_size=2, seed=9)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple GPU access required")
def test_mps_training_scoring_and_cpu_seeded_sampling():
    images = [np.zeros((32, 32, 3), dtype=np.uint8), np.full((32, 32, 3), 180, dtype=np.uint8)]
    cpu = MaskedImageObserver()
    gpu = MaskedImageObserver(device="mps")
    for observer in (cpu, gpu):
        observer.initialize(7)
    assert cpu.state_hash() == gpu.state_hash()
    before = gpu.state_hash()
    errors, predictions = gpu.predict(images)
    assert gpu.state_hash() == before  # scoring must not update BatchNorm
    assert np.isfinite(errors).all() and np.isfinite(predictions).all()
    records = [observer.train([observer.prepare(i) for i in images], steps=2, batch_size=2, seed=9)
               for observer in (cpu, gpu)]
    assert records[0]['sample_sha256'] == records[1]['sample_sha256']
    assert gpu.state_hash() != before
    assert gpu.describe()['device'] == 'mps'
    assert next(gpu.model.parameters()).device.type == 'mps'
