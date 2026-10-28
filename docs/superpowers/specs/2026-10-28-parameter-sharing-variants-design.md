# Parameter-Sharing Variants Design

## Intent
Add opt-in parameter-sharing variants for MASAC and MATD3 without changing the current canonical behavior. The purpose is to benchmark whether homogeneous six-UAV training can reduce model/optimizer duplication and PQL learner backpressure while preserving decentralized execution and the existing global team reward.

## Goals
- A: default/current behavior remains six independent actors + six per-agent TwinCritics.
- B: one shared, agent-conditioned actor + existing six per-agent TwinCritics.
- C: existing six actors + one shared centralized team TwinCritic.
- D: one shared, agent-conditioned actor + one shared centralized team TwinCritic.
- Support MASAC and MATD3, BASE and PQL paths, CPU/CUDA, FP32 and existing optional AMP/compile paths.
- Preserve actor locality: shared actor input is local observation plus static one-hot UAV identity only.
- Preserve the existing scalar team reward and centralized global state.
- Keep old checkpoints loadable when flags are absent/default false; shared variants round-trip their own checkpoints exactly.

## Non-goals
- Do not change environment observations, reward, replay schema, scenario generation, or communication model.
- Do not change current canonical YAML defaults.
- Do not convert or resume an independent-network checkpoint into a shared-network variant.
- Do not claim shared variants are better until benchmark/evaluation data exists.

## Architecture
Two opt-in booleans live in `OffPolicyConfig`: `share_actor_parameters` and `share_critic_parameters`, both default `False` and therefore inherited by MASAC/MATD3.

### Shared actor
A shared actor is stored once in `actors`. Its network input dimension is `observation_dim + num_uavs`. Every call appends a one-hot agent identity to that UAV's local observation. The environment observation itself is unchanged. One actor optimizer owns the shared parameters; per-agent losses are averaged before the single optimizer step so gradient scale remains comparable to one actor loss.

### Shared critic
A shared critic is stored once in `critics` and consumes the same centralized state plus the six-UAV joint action. It represents a team Q. For MATD3, the TD target is directly the shared target TwinCritic value. During actor updates, each agent changes only its own action while peer actions are detached and the same team critic supplies the gradient.

For MASAC, actor losses retain coordinate-wise expected-Q over each UAV's discrete destination while using the same shared critic. The shared critic target is the mean of the six coordinate soft values, keeping reward/Q scale comparable to the current per-agent targets while incorporating each UAV's entropy term. MASAC keeps six continuous and six discrete alpha parameters so entropy adaptation remains agent-specific.

### PQL/checkpoints
PQL snapshots already serialize `actors` and critic state dictionaries, so one-element ModuleLists remain compatible with the worker protocol. Optimizer checkpoint counts follow the number of actual actor optimizers. Saved config records the sharing flags, so shared checkpoints reconstruct the correct topology. Old configs omit the new fields and retain defaults `False`.

## Variants
- A independent: actor=False, critic=False.
- B shared actor: actor=True, critic=False.
- C shared critic: actor=False, critic=True.
- D fully shared: actor=True, critic=True.

## Safety / fairness
Existing training jobs and canonical config are untouched. New runs must tag/name their sharing variant and should be evaluated as a separate ablation before changing the canonical cohort.
