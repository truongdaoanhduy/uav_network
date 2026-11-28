"""Explicit shield-learning tuple variants; raw is the shielded-environment baseline."""
import torch
import warp as wp
from uav_network.algorithms.core.action_types import Action

def store_transitions(replay, old, state, action, rew, nextobs, nextstate, term,
                      world, params, learner_device, mode, lam, *, return_details=False):
    extra=0
    cost=torch.zeros_like(rew)
    training_reward=rew
    # Measure the same geometric correction in every mode, including raw.
    nominal=wp.to_torch(world.buffers["candidates"]).to(learner_device,dtype=torch.float32)
    safe=wp.to_torch(world.buffers["v_safe"]).to(learner_device,dtype=torch.float32)
    failed=wp.to_torch(world.buffers["failed"]).to(learner_device).bool()
    alive=(old[...,9]>.5).unsqueeze(-1)
    delta=(nominal-safe)/float(params.v_max_mps)
    cost=torch.where(failed,torch.zeros_like(rew),(delta.square()*alive).sum((-1,-2)))
    if mode=="raw":
        replay.add_batch(old,state,action,rew,nextobs,nextstate,term)
    else:
        training_reward=rew-float(lam)*cost
        if mode=="raw_penalized":
            replay.add_batch(old,state,action,training_reward,nextobs,nextstate,term)
        else:
            # A successful/depleted terminal step may still commit motion.
            # Only physical failure freezes the proposal instead of committing it.
            executed=(safe/float(params.v_max_mps)).clamp(-1.,1.)
            executed=torch.where(failed[:,None,None],action.continuous,executed)
            exec_action=Action(executed,action.destination)
            intervention=wp.to_torch(world.buffers["intervention"]).to(learner_device).clone()
            # Relabelled safe tuples remove the existing raw-proposal correction
            # cost. This remains an experimental physical-action approximation;
            # no projection-idempotence or unbiased-policy-gradient claim is made.
            safe_reward=rew+float(getattr(params,"w_blocked_motion",0.))*intervention*(~failed)
            replay.add_batch(old,state,exec_action,safe_reward,nextobs,nextstate,term)
            if mode=="dual":
                indices=torch.nonzero(cost>1e-6,as_tuple=False).squeeze(-1)
                extra=int(indices.numel())
                if extra:
                    raw_sub=Action(action.continuous[indices],action.destination[indices])
                    replay.add_batch(old[indices],state[indices],raw_sub,training_reward[indices],
                                     nextobs[indices],nextstate[indices],term[indices])
    details=dict(extra_rows=extra,training_reward=training_reward,shield_cost=cost)
    return details if return_details else extra
