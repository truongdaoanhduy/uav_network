# Shared Actor + Critic FULLGPU Design

## Intent
Add a production-ready FULLGPU counterpart to the existing shared-actor/shared-critic HYBRID75 profile for MASAC and MATD3, while preserving the HYBRID profile for controlled comparison.

## Chosen approach
Keep `configs/comparison-20k-hybrid75-1kb-shared-actor-critic.yaml` unchanged as the HYBRID75 control. Add `configs/comparison-20k-fullgpu-1kb-shared-actor-critic.yaml` that differs only in replay placement and identity metadata: `runtime.replay_device=cuda`, `runtime.replay_cpu_fraction=0`, FULLGPU output/group/tag naming.

## Constraints
- MASAC and MATD3 remain `share_actor_parameters=true` and `share_critic_parameters=true`.
- Preserve seed44, 20T/15O, 1200 steps, 20k episodes, E4096, B8192, replay1M, updates_per_step8, FP32, compile on, BASE, link-budget +3dB.
- Do not launch cloud training in this task.
- HYBRID75 profile remains available for A/B convergence comparison.
