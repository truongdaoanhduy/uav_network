# Reference MASAC / MATD3 Baseline Provenance

This document defines what this repository is allowed to call the **reference baseline**.
The reference profile is:

`configs/comparison-reference-masac-matd3-xyz-sensing.yaml`

The UAV task has a masked hybrid action `(continuous XYZ + RF power, discrete destination)`.
No single vanilla SAC/TD3 library implements exactly this simultaneous action space, so the
baseline is **source-faithful algorithm core plus a documented task-interface adapter**. It
must not be described as byte-for-byte vanilla SAC or TD3.

## Shared execution variant — approved 2026-11-06

The user explicitly requested shared actors/critics and real chunk2048 for both
learners. The canonical XYZ profile now selects those values for MASAC/MATD3.
This is the declared shared-parameter, hidden-LayerNorm reference variant;
sharing changes the architecture and is not an original-paper default.

Both learners bound target, actor-through-critic and regression evaluations by
original replay rows, before agent/destination expansion. Non-reentrant PyTorch
checkpointing recomputes only pure critic functions, with no policy/noise draws
inside them. Full-batch losses, one optimizer step, clipping and TD3 delay remain.
Chunk0 still means full batch. Dataclass defaults and old configs are unchanged.

For B8192/chunk2048 there are4 original-row partitions. A shared MASAC call may
expand2048*6*7=86016 critic rows; MATD3 actor values expand2048*6=12288 rows.
The same original-row limit does not mean equal peak VRAM between algorithms.
This bounds critic activations, not the entire replay/actor/environment memory.

Historical1024 proposal YAMLs and their dated audit remain archived proposals.
The canonical XYZ YAML above is the selected current recipe. PQL is an explicit
execution override, not a modification of SAC/TD3 objectives.

## MASAC extension — approved 2026-11-05

The user subsequently requested the same hidden LayerNorm architecture for
MASAC. The current reference XYZ profile now enables both
`masac.actor_layer_norm=true` and `matd3.actor_layer_norm=true`.
Both actors reuse the same optional encoder construction: two256-unit
`Linear -> nn.LayerNorm -> ReLU` blocks. The MASAC Gaussian mean/log_std and
categorical heads, tanh transformation, log-Jacobian, entropy objectives and
critics are unchanged. BASE/PQL policy and collector copies receive the setting
through their resolved configurations. The MATD3-only section below records
its earlier approval; its statement that MASAC was unchanged is historical.

The shared config option defaults to false. Missing fields in old configurations
and resume snapshots receive that default, keeping the old architecture.
Best-state validation accepts historical config hashes with either disabled
flag omitted, including the intermediate MATD3-only schema. A changed active
actor architecture remains a provenance mismatch. Old policies are loaded
with their original structure; fresh training is required for the new actor.

LayerNorm standardizes hidden features and applies learnable scale/bias;
it does not restrict features to[-1,1]. ReLU then removes negative hidden
activations. Subsequent Linear weights can be negative, so nonnegative hidden
activations do not restrict the policy to positive actions. This order does
not guarantee an unsaturated final tanh or improved convergence.

## MATD3 actor LayerNorm architecture — approved 2026-11-05

The current reference XYZ profile explicitly enables `matd3.actor_layer_norm=true`.
This is a declared architecture variant of the source-backed MATD3 core, not an
exact reproduction of the original author network. Each of the two 256-unit
actor hidden layers is `Linear -> nn.LayerNorm(256) -> ReLU`, using standard
PyTorch defaults (`eps=1e-5`, learnable affine scale/bias). Actor targets,
collector copies and BASE/PQL policy workers use the same resolved setting.
Continuous and categorical output heads are unchanged;
`matd3.output_normalization=false` remains in effect. Critics and MASAC are
unchanged. The layer normalizes each observation independently over hidden
features, preserving decentralized inference.

`MATD3Config.actor_layer_norm` defaults to false for historical configurations
and missing-field checkpoint inference. New checkpoint configs serialize the
true setting so loading rebuilds the exact new architecture. Missing/false
settings preserve historical best-state identities and accepted config hashes;
turning LayerNorm on still rejects a mismatched MATD3 best-state resume. A policy trained
without these layers must retain its original architecture; enabling LayerNorm
is a fresh architecture experiment, not an automatic repair of old weights.
LayerNorm does not mathematically bound the final Linear head, so finite
gradients in short checks are not a full-training convergence claim.

Primary component reference: https://docs.pytorch.org/docs/2.11/generated/torch.nn.LayerNorm.html.
TorchRL MLP supports optional standard normalization via `norm_class`:
https://docs.pytorch.org/rl/0.9/reference/generated/torchrl.modules.MLP.html.
These sources justify the component/interface, not a claim that LayerNorm is
an original default of TD3 or a proven performance fix in this UAV task.

## Primary references

### SAC / MASAC core
- Haarnoja et al., *Soft Actor-Critic Algorithms and Applications*, arXiv:1812.05905.
- Stable-Baselines3 SAC source: https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/sac
- BenchMARL MASAC source/config: https://github.com/facebookresearch/BenchMARL/tree/main/benchmarl/algorithms and `benchmarl/conf/algorithm/masac.yaml`.
- Delalleau et al., *Discrete and Continuous Action Representation for Practical RL in Video Games*, arXiv:1912.11077, for Hybrid SAC with simultaneous discrete/continuous components, exact discrete expectation, and separate entropy weights.
- Fu et al., *Multi-agent Reinforcement Learning with Hybrid Action Space for Free Gait Motion Planning of Hexapod Robots*, CoRL2024 / PMLR270 (2025), https://proceedings.mlr.press/v270/fu25c.html, for the CTDE Hybrid-MASAC per-agent critic/actor/temperature objectives. The reference baseline does **not** claim to reproduce Fu et al.'s attention-critic architecture.
- Hua et al., *A further exploration of deep Multi-Agent Reinforcement Learning with Hybrid Action Space*, arXiv:2208.14447, as an additional CTDE hybrid-action MARL reference.

### TD3 / MATD3 core
- Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic Methods*, ICML 2018 / arXiv:1802.09477.
- Original TD3 source: https://github.com/sfujim/TD3
- Ackermann et al., *Reducing Overestimation Bias in Multi-Agent Domains Using Double Centralized Critics*, arXiv:1910.01465.
- Authors' MATD3 source: https://github.com/JohannesAck/MATD3implementation

## Source hierarchy
1. The published algorithm equations define the baseline.
2. Author-released or established library code supplies implementation details the paper leaves unspecified.
3. UAV-specific adaptations are permitted only at the environment/action interface and are listed below.
4. A source-only heuristic that is absent from the paper objective is not copied merely because it exists in a repository. For example, the Ackermann code's small regularizer on policy distribution parameters/logits (not neural-network weight decay) is not part of the MATD3 paper objective, so the reference actor loss remains `-Q1`.

## Source-backed MASAC contract
- off-policy replay training;
- stochastic tanh-squashed Gaussian continuous policy;
- the simultaneous mixed-action actor has a categorical destination branch and a Gaussian continuous branch from the local observation, following Delalleau et al.; the centralized critic evaluates each discrete alternative exactly while continuous actions are sampled;
- the critic uses the multi-pass form (discrete choice supplied/evaluated as an input alternative) rather than Delalleau's preferred all-discrete-output head; Delalleau explicitly identifies multi-pass as a valid alternative, so this is a source-backed architecture choice rather than a UAV performance tweak;
- SAC log-std bounds `[-20, 2]`;
- two Q functions and target Q networks, using the minimum Q estimate;
- learned log-temperature, with `alpha_c[i]` and `alpha_d[i]` per agent, matching the Hybrid-MASAC temperature objectives;
- no hard min/max alpha clamp in the reference path;
- nominal continuous auto-entropy target `-dim(A_continuous)` (`-4` for XYZ + learned RF power); the UAV adapter removes silent RF and inactive controls, so the effective target for an active XYZ agent is `-E[3 + 1(non-silent)]` and is zero for an inactive agent;
- discrete auto-entropy target `0.2 * log(total_destination_count)`, matching BenchMARL/TorchRL's fixed `num_actions` auto-target; the action mask changes sampling support but does not redefine the reference target;
- no entropy-target annealing in the reference profile;
- critic regression uses the SAC/Hybrid-MASAC `0.5 * squared TD error` per Q network and preserves a separate batch-mean objective for each independent agent critic;
- no gradient clipping, matching the Hybrid SAC implementation description from Delalleau et al.;
- centralized critics and decentralized local policies.

## Source-backed MATD3 contract
- deterministic tanh actor for the continuous branch;
- two centralized critics per agent and the minimum target Q;
- delayed actor/target update (`policy_delay=2`);
- target-policy Gaussian smoothing (`std=0.2`, clipped to `0.5`) on continuous controls;
- Gaussian continuous exploration noise (`std=0.1`);
- actor loss through critic 1;
- during an agent actor update, peer actions come from the replay minibatch and only the optimized agent's action is replaced by its current policy action, matching the released MATD3 implementation;
- differentiable categorical action uses a soft Gumbel-Softmax sample as in the released MATD3/MADDPG distribution code;
- no custom pre-tanh vector output normalization;
- gradient clipping follows the released MATD3 implementation: each parameter tensor gradient is clipped independently to norm `0.5`, rather than using the repository-wide global-norm `10` heuristic; independent critic losses retain a batch-only mean before this clipping (the agent average is undone).

## Required UAV interface adaptations
These are task semantics, not claimed algorithm improvements:
- invalid communication destinations are masked before categorical sampling;
- the discrete destination is represented as one-hot/relaxed-one-hot inside the critic;
- the environment boundary converts the sampled destination to an integer;
- RF power is zero when the destination is silent and motion is zero for inactive UAVs;
- critics consume the centralized global state and joint UAV action;
- for MASAC only, Q is evaluated over each own-destination alternative and probability-weighted exactly, with invalid categories having zero probability; this is the multi-agent form of Hybrid SAC's finite discrete expectation while peer actions are detached/sampled from their policies;
- MATD3 replay stores hard one-hot destinations from the integer environment boundary; its actor and target critic inputs use soft Gumbel samples, with RF power gated by the sampled category's hard argmax. The original MPE author implementation stores its soft action vectors in replay. This hard-replay/soft-training interface is therefore a declared UAV adaptation, not identical released-author behavior;
- the cooperative team reward is supplied by the environment.

The reference profile deliberately keeps the discrete entropy target tied to the fixed total
destination count, because BenchMARL/TorchRL passes `num_actions` to `DiscreteSACLoss` even
when the actor uses a mask. Historical UAV variants that retarget entropy to the number of
currently valid destinations remain available for ablation/backward compatibility, but that
mask-aware retargeting is not part of the reference baseline. The value `0.2` is from the
BenchMARL MASAC configuration, not a universal TorchRL default.

Clipping telemetry follows the chosen rule: `critic_grad_norm` is the global gradient norm
for global/none mode and the maximum individual tensor norm for per-parameter mode.
`critic_gradient_clipped` is always false for none mode, and uses the corresponding
pre-clip norm for other modes. This avoids reporting a global-norm crossing as a
per-variable clipping event.

## Deliberately NOT active in the reference profile
- UAV-specific discrete entropy annealing `0.9 -> 0.5`;
- mask-cardinality-aware discrete entropy retargeting;
- hard alpha clamp to `exp(5)`;
- MATD3 vector output normalization;
- epsilon-greedy discrete destination schedule;
- hard-argmax/softmax straight-through estimator for MATD3 categorical gradients;
- current-policy peer substitution in the MATD3 actor loss;
- `torch.compile` and critic vectorization;
- AMP/fused optimizers;
- PQL execution mode (opt-in comparison override).

The disabled stabilizers above remain outside this source-backed recipe.
The explicitly declared shared/LN/chunk execution variant is now authorized;
it must not be described as a literal reproduction of an upstream network.

## Common experiment settings are not algorithm claims
The UAV scenario, reward, CBF safety layer, sensing model, network simulator, number of
environments, replay capacity, batch size, update-to-data ratio, seed and wall-clock budget
are experiment settings. Learning rate, discount, Polyak coefficient and network width are
also study hyperparameters unless a result explicitly claims to reproduce a source paper's
experiment table. They may be controlled or tuned under a declared protocol, but they must
not be described as defaults from the SAC/TD3 papers and must not be changed post-hoc to
make one algorithm look better or worse.

## Reproducibility rule
A result may be labelled `REFERENCE_BASELINE` only if its resolved config passes
`tests/test_reference_baseline_contract.py`. Any algorithmic modification must use a new
variant name and must not silently replace this profile.

## MASAC MASKAWARE

The fixed-target reference YAML remains available as a control. The separate
MASAC MASKAWARE variant uses the same YAML with
`--set masac.mask_aware_discrete_entropy_target=true`. For each observation it
sets the discrete target to `0.2 * log(K)`, where `K` counts valid destinations
including silence. If a UAV has only silence across its batch, its discrete
temperature stays at its previous value. This addresses the unattainable
positive target on forced-silence transitions observed in run `2734dbf3`.

The Kaggle UI notebooks apply the override for both BASE and PQL and include
`MASKAWARE` in their W&B run names. To reproduce their NOCHUNK settings, also
set `runtime.algorithm=masac` and `runtime.critic_chunk_size=0`, with
`pql.enabled=false` for BASE or `pql.enabled=true` for PQL. Use a fresh
checkpoint lineage for each variant. Evaluate against the fixed-target
control on the same scenarios, seeds, and environment-step budgets. Report
the override alongside delivery, search, and return.

## Verification sources inspected on 2026-11-03
- Ackermann author source revision: `fd6c7d0df4fc4effd4e0fa11abdbc7d12d8f01a1`.
- `matd3/matd3/trainer/matd3.py`: replay peers, independent batch-mean critics, `0.5` clipping.
- Haarnoja/SB3 SAC critic source: each Q regression carries the canonical `0.5 * MSE` factor.
- `matd3/common/distributions.py`: soft categorical sample; `matd3/common/tf_util.py`: per-variable clip.
- Fujimoto `sfujim/TD3/TD3.py`: tanh actor, twin target minimum, delay and Gaussian smoothing.
- BenchMARL `benchmarl/conf/algorithm/masac.yaml`: alpha1, discrete target0.2, no alpha bounds.
- Delalleau arXiv:1912.11077 Appendix: Hybrid SAC does not use gradient clipping.
- Existing source-contract tests pin this implementation; they do not prove numerical
  identity to a whole upstream repository. The independent optimizer oracle tests the
  MATD3 per-agent mean/clipping/Adam update against separately optimized source equations.

Historical study YAMLs are not automatically migrated: some retain entropy
annealing or old output normalization. Select the explicit reference YAML above
for the current declared shared/LN/chunk variant; launchers and old runs are not
silently switched. Serialized
legacy MATD3 epsilon fields remain for old config/checkpoint compatibility and do
not control the source-backed categorical sampler.
