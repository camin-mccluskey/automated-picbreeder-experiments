"""Learnable scalar fixture and device-local replay, without global RNG changes."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from automated_picbreeder.offspring_prediction import OffspringValuePredictor


@pytest.mark.parametrize('device', ['cpu', 'mps'])
@pytest.mark.parametrize('value_source,context_size', [('predictability', 23), ('imagenet', 21)])
def test_predictor_learns_and_replays_without_touching_global_rng(device, value_source, context_size):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        state = torch.random.get_rng_state().clone()
        model = OffspringValuePredictor(device=device, value_source=value_source)
        model.initialize(7)
        images = [np.full((16,16,3), v, np.uint8) for v in (0,255)]
        inputs = model.prepare(images, np.full((16,16,3), .5), np.zeros((2,context_size)))
        labels = np.array([.2,.8])
        before = model.predict(inputs)
        initial_hash = model.state_hash()
        assert initial_hash == model.state_hash()
        replay = [{'inputs': x, 'target': y} for x,y in zip(inputs,labels)]
        update = model.train(replay, steps=80, batch_size=8, seed=19)
        after = model.predict(inputs)
        assert np.mean((after-labels)**2) < np.mean((before-labels)**2)*.1
        assert model.state_hash() != initial_hash
        repeated = OffspringValuePredictor(device=device, value_source=value_source)
        repeated.initialize(7)
        assert repeated.state_hash() == initial_hash
        assert repeated.train(replay, steps=80, batch_size=8, seed=19) == update
        np.testing.assert_array_equal(repeated.predict(inputs), after)
        assert repeated.state_hash() == model.state_hash()
        assert torch.equal(state, torch.random.get_rng_state())
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize('aggregation', ['max', 'mean'])
@pytest.mark.parametrize('value_source,context_size', [('predictability', 23), ('imagenet', 21)])
def test_aggregation_is_run_configuration_without_altering_predictor_inputs(aggregation, value_source, context_size):
    model = OffspringValuePredictor(offspring_aggregation=aggregation, value_source=value_source)
    assert model.context_size == context_size
    metadata = model.describe()
    assert metadata['offspring_aggregation'] == aggregation
    assert aggregation in metadata['loss']
    assert 'previous-grid RGB mean' not in metadata['inputs']
    reference_image = np.full((8, 8, 3), .25)
    images = [np.full((8, 8, 3), 255, dtype=np.uint8)]
    inputs = model.prepare(images, reference_image=reference_image, contexts=np.zeros((1, context_size)))
    reference_image[:] = 0
    np.testing.assert_array_equal(inputs[0][0][:3].numpy(), np.ones((3, 8, 8)))
    np.testing.assert_array_equal(inputs[0][0][3:].numpy(), np.full((3, 8, 8), .25))


def test_predictor_defaults_to_max():
    assert OffspringValuePredictor().describe()['offspring_aggregation'] == 'max'


@pytest.mark.parametrize('aggregation', ['median', '', None, 1, True])
def test_predictor_rejects_invalid_aggregation(aggregation):
    with pytest.raises(ValueError, match='offspring_aggregation'):
        OffspringValuePredictor(offspring_aggregation=aggregation)
