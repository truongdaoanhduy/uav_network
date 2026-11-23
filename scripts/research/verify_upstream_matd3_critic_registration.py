"""Non-mutating probe of a pinned public MATD3 repository's registered Q heads.

Checks only whether the two public-source output heads are registered with Adam.
No code copied into or changed inside the user's algorithm implementation.
Usage:
 python scripts/research/verify_upstream_matd3_critic_registration.py \
   --upstream outputs/reference_source_audit_20261108/MATD3pytorch
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def check(upstream):
    import gymnasium as gym
    sys.modules.setdefault("gym", gym)
    sys.modules.setdefault("gym.spaces", gym.spaces)
    sys.path.insert(0, str(upstream.resolve()))
    import torch
    from types import SimpleNamespace
    from offpolicy.algorithms.matd3.algorithm.actor_critic import MATD3_Critic

    torch.manual_seed(44)
    args = SimpleNamespace(
        use_orthogonal=False, hidden_size=16,
        use_feature_normalization=False, use_ReLU=True,
        use_conv1d=False, stacked_frames=1, layer_N=1,
    )
    model = MATD3_Critic(args, central_obs_dim=8,
                         central_act_dim=4, device=torch.device("cpu"))
    first = [layer.weight.detach().clone() for layer in model.q_outs]
    registered = {id(p) for p in model.parameters()}
    registered_before = [id(head.weight) in registered for head in model.q_outs]
    inputs = (torch.randn(5, 8), torch.randn(5, 4))
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        optim.zero_grad()
        qs = model(*inputs)
        loss = sum(q.square().mean() for q in qs)
        loss.backward()
        optim.step()
    weights_changed = [not torch.equal(head.weight.detach(), original)
                       for head, original in zip(model.q_outs, first)]
    gradients_exist = [head.weight.grad is not None for head in model.q_outs]
    # Demonstrate the issue is exactly submodule registration, not data or loss.
    # This changes a temporary in-memory copy; never patch upstream source files.
    from copy import deepcopy
    repaired = deepcopy(model)
    repaired.q_outs = torch.nn.ModuleList(repaired.q_outs)
    repaired_before = [head.weight.detach().clone() for head in repaired.q_outs]
    repaired_optimizer = torch.optim.Adam(repaired.parameters(), lr=1e-3)
    for _ in range(3):
        repaired_optimizer.zero_grad()
        values = repaired(*inputs)
        repaired_loss = sum(q.square().mean() for q in values)
        repaired_loss.backward()
        repaired_optimizer.step()
    registration_repair = {
        "q_head_registered": [
            id(head.weight) in {id(param) for param in repaired.parameters()}
            for head in repaired.q_outs
        ],
        "q_head_weights_changed_after_3_adam_steps": [
            not torch.equal(head.weight.detach(), weight)
            for head,weight in zip(repaired.q_outs,repaired_before)
        ],
        "scope": "temporary in-memory ModuleList replacement; upstream files unchanged",
    }
    sha = subprocess.check_output(["git", "-C", str(upstream),
                                  "rev-parse", "HEAD"], text=True).strip()
    return {
        "upstream": "marlbenchmark/off-policy",
        "upstream_git_commit": sha,
        "source": "offpolicy/algorithms/maddpg/algorithm/actor_critic.py",
        "q_heads": len(model.q_outs),
        "q_head_registered": registered_before,
        "q_head_grad_exists": gradients_exist,
        "q_head_weights_changed_after_3_adam_steps": weights_changed,
        "in_memory_modulelist_repair": registration_repair,
        "finding": (
            "Unregistered Q output heads: upstream Python list bypasses nn.Module registration"
            if not all(registered_before) else "All upstream Q output heads registered"
        ),
        "scope": "Pinned upstream source only; not a result about the UAV user's own MATD3.",
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream",type=Path,
                   default=Path("outputs/reference_source_audit_20261108/MATD3pytorch"))
    p.add_argument("--out",type=Path,
                   default=Path("docs/methods/evidence/reference_matd3_q_registration_2026-11-08.json"))
    a=p.parse_args()
    result=check(a.upstream)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if result["upstream_git_commit"].startswith("41fd5eb"):
        assert result["q_head_registered"]==[False,False]
        assert result["q_head_grad_exists"]==[True,True]
        assert result["q_head_weights_changed_after_3_adam_steps"]==[False,False]
        assert result["in_memory_modulelist_repair"]["q_head_registered"]==[True,True]
        assert result["in_memory_modulelist_repair"]["q_head_weights_changed_after_3_adam_steps"]==[True,True]


if __name__=="__main__":
    main()
