"""Batch independent critics without replacing their parameters or Adam state."""
import torch
from torch.func import functional_call


def critic_values(critics, state, actions, vectorized=False, per_critic=False):
    """Return twin values [batch, agent]; optional actions [agent,batch,N,A]."""
    if not vectorized:
        values = [critic(state, actions[i] if per_critic else actions)
                  for i, critic in enumerate(critics)]
        return tuple(torch.stack([value[k] for value in values], 1) for k in (0, 1))
    # torch.stack retains paths to the original leaves and reads current weights
    # every call. stack_module_state instead creates unrelated new leaf tensors.
    states = [dict(critic.named_parameters()) for critic in critics]
    parameters = {name: torch.stack([p[name] for p in states]) for name in states[0]}

    def evaluate(params, local_actions):
        return functional_call(critics[0], params, (state, local_actions), strict=True)

    values = torch.vmap(evaluate, in_dims=(0, 0 if per_critic else None))(parameters, actions)
    return tuple(value.transpose(0, 1).contiguous() for value in values)
