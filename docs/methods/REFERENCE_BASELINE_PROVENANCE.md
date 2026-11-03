# Reference MASAC / MATD3 Baseline Provenance

This document defines what this repository is allowed to call the **reference baseline**.
The reference profile is:

`configs/comparison-reference-masac-matd3-xyz-sensing.yaml`

The UAV task has a masked hybrid action `(continuous XYZ + RF power, discrete destination)`.
No single vanilla SAC/TD3 library implements exactly this simultaneous action space, so the
baseline is **source-faithful algorithm core plus a documented task-interface adapter**. It
must not be described as byte-for-byte vanilla SAC or TD3.

## Primary references

### SAC / MASAC core
- Haarnoja et al., *Soft Actor-Critic Algorithms and Applications*, arXiv:1812.05905.
- Stable-Baselines3 SAC source: https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/sac
- BenchMARL MASAC source/config: https://github.com/facebookresearch/BenchMARL/tree/main/benchmarl/algorithms and `benchmarl/conf/algorithm/masac.yaml`.
- Delalleau et al., *Discrete and Continuous Action Representation for Practical RL in Video Games*, arXiv:1912.11077, for Hybrid SAC with simultaneous discrete/continuous components, exact discrete expectation, and separate entropy weights.
- Fu et al., *Multi-agent Reinforcement Learning with Hybrid Action Space for Free Gait Motion Planning of Hexapod Robots*, CoRL2024 / PMLR270 (2025), https://proceedings.mlr.press/v270/fu25c.html, for the CTDE Hybrid-MASAC extension and per-agent continuous/discrete temperatures.
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
4. A source-only heuristic that is absent from the paper objective is not copied merely because it exists in a repository. For example, the Ackermann code's small actor parameter regularizer is not part of the MATD3 paper objective, so the reference actor loss remains `-Q1`.

## Source-backed MASAC contract
- off-policy replay training;
- stochastic tanh-squashed Gaussian continuous policy;
- the simultaneous mixed-action actor has a categorical destination branch and a Gaussian continuous branch from the local observation, following Delalleau et al.; the centralized critic evaluates each discrete alternative exactly while continuous actions are sampled;
- SAC log-std bounds `[-20, 2]`;
- two Q functions and target Q networks, using the minimum Q estimate;
- learned log-temperature, with `alpha_c[i]` and `alpha_d[i]` per agent, matching the Hybrid-MASAC temperature objectives;
- no hard min/max alpha clamp in the reference path;
- nominal continuous auto-entropy target `-dim(A_continuous)` (`-4` for XYZ + learned RF power); the UAV adapter removes silent RF and inactive controls, so the effective target for an active XYZ agent is `-E[3 + 1(non-silent)]` and is zero for an inactive agent;
- discrete auto-entropy weight `0.2 * log(number_of_valid_choices)`, adapting BenchMARL's configured discrete SAC target to a state-dependent validity mask;
- no entropy-target annealing in the reference profile;
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

The mask-aware discrete entropy target uses the number of *currently valid* choices rather
than the fixed total destination count. This is explicitly an interface adaptation required
by the state-dependent action set. The value `0.2` is from the BenchMARL MASAC configuration, not a universal TorchRL default (the inspected TorchRL DiscreteSACLoss default is `0.98`).

Clipping telemetry follows the chosen rule: `critic_grad_norm` is the global gradient norm
for global/none mode and the maximum individual tensor norm for per-parameter mode.
`critic_gradient_clipped` is always false for none mode, and uses the corresponding
pre-clip norm for other modes. This avoids reporting a global-norm crossing as a
per-variable clipping event.

## Deliberately NOT active in the reference profile
- UAV-specific discrete entropy annealing `0.9 -> 0.5`;
- hard alpha clamp to `exp(5)`;
- MATD3 vector output normalization;
- epsilon-greedy discrete destination schedule;
- hard-argmax/softmax straight-through estimator for MATD3 categorical gradients;
- current-policy peer substitution in the MATD3 actor loss;
- critic chunking/checkpointing;
- `torch.compile` and critic vectorization;
- AMP/fused optimizers;
- PQL execution mode;
- shared actor or shared critic parameters.

Those mechanisms may exist in historical or ablation code, but a run using them must not be
reported as the reference baseline.

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

## Verification sources inspected on 2026-11-03
- Ackermann author source revision: `fd6c7d0df4fc4effd4e0fa11abdbc7d12d8f01a1`.
- `matd3/matd3/trainer/matd3.py`: replay peers, independent batch-mean critics, `0.5` clipping.
- `matd3/common/distributions.py`: soft categorical sample; `matd3/common/tf_util.py`: per-variable clip.
- Fujimoto `sfujim/TD3/TD3.py`: tanh actor, twin target minimum, delay and Gaussian smoothing.
- BenchMARL `benchmarl/conf/algorithm/masac.yaml`: alpha1, discrete target0.2, no alpha bounds.
- Delalleau arXiv:1912.11077 Appendix: Hybrid SAC does not use gradient clipping.
- Existing source-contract tests pin this implementation; they do not prove numerical
  identity to a whole upstream repository. The independent optimizer oracle tests the
  MATD3 per-agent mean/clipping/Adam update against separately optimized source equations.

Historical study YAMLs are not the reference profile: some intentionally retain shared
parameters and entropy annealing. Select the explicit reference YAML above to claim
this baseline; no supervisor or historical run was silently migrated. Serialized
legacy MATD3 epsilon fields remain for old config/checkpoint compatibility and do
not control the source-backed categorical sampler.
