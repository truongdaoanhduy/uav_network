"""Chunking must reduce retained activations and preserve the logical MASAC update."""
from copy import deepcopy

import pytest
import torch

from uav_network.configuration import Config, load_config
from uav_network.algorithms.masac import MASAC


def learner(chunk):
    c = Config()
    c.masac.hidden_sizes = [16, 16]
    c.runtime.vectorize_critics = True
    c.runtime.critic_chunk_size = chunk
    return MASAC(c, 12, 5)


def test_primary_enables_bounded_critic_execution_and_rejects_negative_chunks():
    c = load_config("configs/comparison-20k-hybrid75-1kb.yaml")
    assert c.runtime.critic_chunk_size == 1024
    c.runtime.critic_chunk_size = -1
    with pytest.raises(ValueError, match="critic_chunk_size"):
        c.validate()


def expected_q(model, state, joint, continuous, vectorized):
    if vectorized:
        return model._expected_q_all(model.critics, state, joint, continuous)
    return torch.stack([model._expected_q(critic, state, joint, continuous[:, i], i)
                        for i, critic in enumerate(model.critics)], 1)


@pytest.mark.parametrize("vectorized", [False, True])
def test_chunked_destination_values_gradients_rng_and_retained_storage(vectorized):
    torch.set_num_threads(1)
    torch.manual_seed(44)
    reference = learner(0)
    actual = learner(4)
    actual.load_state_dict(reference.state_dict())
    reference.critics.requires_grad_(False)
    actual.critics.requires_grad_(False)
    state = torch.randn(13, 5)
    joint = torch.randn(13, 6, 10)
    continuous = torch.randn(13, 6, 3)
    retained = []
    outputs = []
    gradients = []
    for model in (reference, actual):
        action = continuous.clone().requires_grad_()
        storage = {}
        def pack(tensor):
            backing = tensor.untyped_storage()
            storage[backing.data_ptr()] = backing.nbytes()
            return tensor
        before_rng = torch.random.get_rng_state().clone()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            result = expected_q(model, state, joint, action, vectorized)
        retained.append(sum(storage.values()))
        outputs.append(result.detach().clone())
        result.square().mean().backward()
        gradients.append(action.grad.clone())
        assert torch.equal(torch.random.get_rng_state(), before_rng)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=2e-6, atol=2e-6)
    assert retained[1] < retained[0] * .8


def test_complete_chunked_updates_preserve_sampling_and_optimizer_steps():
    torch.set_num_threads(1)
    torch.manual_seed(72)
    reference = learner(0)
    actual = learner(4)
    actual.load_state_dict(reference.state_dict())
    observation = torch.randn(11, 6, 12)
    observation[:, :, 9] = 1
    observation[:, :, -7:] = 1
    batch = {"observation": observation, "next_observation": observation.clone(),
             "state": torch.randn(11, 5), "next_state": torch.randn(11, 5),
             "continuous": torch.zeros(11, 6, 3),
             "destination": torch.zeros(11, 6, dtype=torch.long),
             "reward": torch.arange(11).float(),
             "terminated": torch.tensor([0., 1.] * 5 + [0.])}
    for step in range(3):
        torch.manual_seed(44 + step)
        expected = reference.update(batch)
        rng = torch.random.get_rng_state().clone()
        torch.manual_seed(44 + step)
        observed = actual.update(batch)
        assert torch.equal(torch.random.get_rng_state(), rng)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[name], value,
                                       rtol=3e-5, atol=3e-5)
        for name, value in expected.items():
            assert observed[name] == pytest.approx(value, rel=3e-5, abs=3e-5)
        assert reference.updates == actual.updates == step + 1
        assert reference.actor_updates == actual.actor_updates == step + 1
