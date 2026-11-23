# Reviewer-grade MASAC baseline correctness audit — 2026-11-08

## Verdict and scope

**No demonstrated error in the mathematical SAC/Hybrid-SAC core examined.**
It would nevertheless be incorrect to claim the MASAC baseline is
*fully proven bug-free*, *identical to Fu et al.'s Hybrid-MASAC*, or
*asymptotically worse than MATD3*. Current evidence points primarily to
**slow learning of a useful deterministic continuous-control policy
and high CBF intervention under the early policy**, with an additional
**cross-runtime deterministic-evaluation replay discrepancy** that remains
unresolved. This audit changes NO MASAC/TD3 learning equations and starts
NO training jobs.

The historical training W&B source commit is
`a6e07a8f4b199028298b6ffd9a47f2cecd654fc9`;
current source is the committed paper-core branch after `e63398a`.
Diff of the algorithm files is docstrings/comments only, while the
production environment and communication code are unchanged in that diff.
See [source-faithfulness audit](PAPER_CORE_VS_EMPIRICAL_BASELINE_35T25O_2026-11-08.md)
for the full equation correspondence.

## What the mathematical inspection establishes — and what it does not

| Contract | Source and finding |
| --- | --- |
| Gaussian action and log-density | `uav_network/algorithms/models.py:30-50`: `Normal(mean,std).rsample()`, `tanh`, stable `tanh` Jacobian. Test-time deterministic action uses `tanh(mean)`. Matches continuous SAC core. |
| Masked categorical routing | `models.py` plus `action_utils.py`: invalid destinations have zero probability; silent index 0 always valid. UAV-specific adaptation, not directly specified by original single-agent SAC. |
| Joint twin critic | `masac.py:25-57,149-201`: centralized state + joint action, two Q values, min-double-Q, finite categorical enumeration. Shared team Q differs from the original per-agent/attention-based formulations and must be disclosed. |
| Soft Bellman target | `masac.py:207-256`: reward + gamma * (1 - terminated) * [expected min target Q + categorical and continuous entropy terms]. Independent per-agent Q uses own agent's entropy; Shared team Q uses summed agent entropies. |
| Policy objective | `masac.py:259-324`: minimize alpha_c log probability + sum over legal destinations p_d(alpha_d log p_d - minQ); randomized peer actions detached. Core hybrid SAC gradient, not a proof of fidelity to all Hybrid-MASAC variants. |
| Two adaptive temperatures | `masac.py:325-352`: original-alpha dual or log-alpha parameterization, with per-agent targets; current 35T25O uses `temperature_loss=alpha`, `target_continuous_entropy=-4` scaled for conditional power, `target_discrete_entropy_ratio=0.2` and masked valid-support targets. Automatic entropy is **not an error** merely because alpha decays. |
| Replay, termination and gradients | `training.py`, `optimization.py`, existing tests: replay contains commanded actions and global state, timeout bootstrap is retained, disabled gradient clipping for MASAC is explicit, actor and destination head receive gradients when destinations exist. |

**Not a literal paper reproduction:** Fu et al. (CoRL 2025) study
hexapod footstep planning and use an architectural setup unlike this
UAV team's hybrid XYZ/RF destination controller. The code is a
**transparent Hybrid-SAC/MASAC adaptation under CTDE**, not
the authors' original attention model. In a paper, specify architecture
and loss equations rather than claiming direct reproduction.

Independent numerical and integration checks:
`tests/test_masac.py`,
`tests/test_reference_critic_gradients.py`,
`tests/test_algorithm_math_regressions.py`,
`tests/test_paper_faithfulness_audit_20261108.py`,
`tests/test_paper_core_35t25o_baseline.py`,
and new `tests/test_masac_policy_mode_diagnostic.py`.
During this audit **69 pytest tests + 3 subtests passed**.
These are important but do NOT rule out unidentified implementation faults.

## Same-checkpoint factorial: what actually makes deterministic MASAC weak?

On two authentic HF best Actor snapshots, whose exact SHA256 digests
match `docs/methods/evidence/wandb_latest_ten_2026-11-08.json`, executed
the same 5 validation layout seeds `1000000044..48` using
`WarpMissionEnv` CPU and **zero training**.
Held checkpoint and scenario constant; separately changed continuous
Gaussian sample versus `tanh(mean)`, and destination sample versus argmax.
One sampled rollout per mode/seed, not multiple RNG replicates.
Source: `scripts/diagnose_masac_policy_modes.py`.
Full per-seed evidence:
`docs/methods/evidence/masac_four_policy_modes_2026-11-08.json`.

| Frozen MASAC Actor | Mean + argmax | Gaussian sample + argmax | Mean + categorical sample | Both sampled |
| --- | ---: | ---: | ---: | ---: |
| H512/U16/ActorLN `c70aa6f4`, ~3.69M training transitions | **18.86%** | **33.14%** | 21.71% | 37.14% |
| H256/U8/ActorLN `7f5fd73f`, ~25.80M training transitions | **57.71%** | 51.43% | 51.43% | 53.71% |

**Finding:** In the early U16 actor, continuous Gaussian sampling
explains most of the gap to the stochastic policy. Merely sampling
destination while keeping the mean movement action adds only
2.86 percentage points. In the later H256 U8 actor, mean+argmax is
already better than the sampled modes in this local CPU probe.

**Causal scope:** These are comparisons of inference-mode settings
*within* one snapshot and the same seed bank, not proof that the
U16-vs-U8 difference was *caused* by more training. Those snapshots
also differ in architecture, and rollout randomness is not repeated.
The result indicates why selecting the mean of an incompletely
trained stochastic movement policy can underperform a sampled action.
It does **not** validate changing the benchmark to stochastic-only
evaluation to improve the reported score: deterministic evaluation
is the standard SAC exploitation policy (Spinning Up).

## W&B controlled evidence: the early MASAC policy receives frequent CBF correction

At the matched ~7.37M-global-transition checkpoint, Shared+ActorLN/U8:

| Width | Algorithm | Validation delivery | CBF interventions per UAV-step |
| --- | --- | ---: | ---: |
| H256 | MASAC `7f5fd73f` | 4.57% | 0.93224 |
| H256 | MATD3 `635d7134` | 40.57% | 0.61048 |
| H512 | MASAC `d9c1c5e9` | 3.43% | 0.97667 |
| H512 | MATD3 `ccaadcd6` | 25.14% | 0.80424 |

For MASAC H256/U8, by its later W&B best checkpoint (~25.80M)
delivery improves to **52.00%** and CBF metric to **0.67430**.
MATD3 H256/U8's W&B best is **51.43%** at ~12.90M and CBF metric
**0.59558**. MASAC H512/U8's best is **46.29%** at ~14.75M; its
training did not finish the full 20k episode budget. MATD3 H512/U8's
best **58.86%** is from ~12.90M. These best values have **unequal
checkpoints** and cannot be called matched-budget final rankings.

**Interpretation:** the early MASAC deterministic policy both searches
poorly and requires extensive shield assistance. This may be a
bad movement policy causing CBF correction, a highly intrusive shield
interacting with policy learning, or both. The positive
correlation between decreasing intervention and better delivery
does NOT establish causal direction. Large masks do not by
themselves establish a routing bug: the Actor is allowed only to
send when a valid receiver exists. In the local factorial probe
only roughly 0.4%-1.2% of observed UAV-states even had more than
one legal destination, varying with visited states and mode.

MASAC U8 search versus delivery at matched validation:
**5.14% search / 4.57% delivered**. At later best:
**54.29% search / 52% delivered**. The key weakness appears in
search/navigation rather than a massive gap between discovered and
delivered targets.

### Different update budgets and benchmark fairness

Both baselines used 4096 environment instances, batch8192, 8 updates
per collection, hybrid replay, seed44 and their corresponding paired
architecture. However MATD3 delays Actor updates to every second
Critic update, whereas MASAC updates the Actor (and temperature) on
every step. This affects compute cost and convergence even at the same
global environment transitions. The source choices for temperature,
gradient clipping and policy stochasticity are also different by
algorithm. A fair paper should state both environment-step sample
efficiency and wall-clock/runtime cost, without making the
unjustified claim that equal network depth means equal learning
dynamics. A single training seed and five evaluation layouts are
not statistically equivalent to multiple independent training seeds.

## Important reproducibility warning: exact W&B best snapshot replays differently

This audit *also downloaded*, read-only, the original W&B per-seed
`evaluations/eval_step_000025804800_policy_50144/metrics.json`
from run `7f5fd73f`, and compared against the SHA-verified frozen
HF best snapshot and current official CPU evaluator.

**Original W&B delivery: 52.00% (91/175 target opportunities).**
**Current local CPU replay of the same checkpoint: 57.71% (101/175).**
All **five initial environment layout SHA256 hashes match exactly**
with the original W&B records, as do the seed list and checkpoint
identity. **Final steps and trajectories diverge**.

Evidence:
`docs/methods/evidence/masac_wandb_local_same_seed_replay_2026-11-08.json`.

This rules out *different initial scenes* as the cause of that
particular discrepancy. It does **not** establish the cause:
numerical/runtime differences, nondeterministic kernels or an
unrecorded difference in the inference/evaluation execution
environment remain hypotheses. Torch deterministic mode does
not automatically guarantee cross-machine bitwise-identical
long-horizon trajectories. **Do not replace original W&B values
with local replay values or claim exact end-to-end reproducibility
until this is resolved.** Both have separately recorded provenance.
Original validation provenance is the W&B event and checkpoint.
The current 2x2 analysis is explicitly labeled a separate CPU
experiment rather than a redefinition of the historical score.

## Reviewer-facing verdict and the one next task

1. **Equation/source audit: pass for the checked SAC/hybrid terms,
   not a full author-paper-identical MASAC implementation**.
2. **Empirical explanation for early low MASAC: strong evidence**:
   poor deterministic *movement* policy plus many CBF
   interventions; categorical routing sampling is secondary.
3. **Not yet established**: that MATD3 intrinsically beats
   MASAC in this task, that MASAC is bug-free across all paths,
   or that the CBF is causally responsible for the gap.
4. **Highest priority unresolved technical task (read-only)**:
   reproduce the exact saved best MASAC checkpoint in a
   pinned evaluator container matching Kaggle's runtime, with
   controlled Torch/Warp versions and recorded first 10
   actions/positions/CBF decisions; find the first step
   of divergence from original evaluation, or explicitly
   document the cross-platform determinism limitation.
   Do not launch another costly training sweep to debug
   an evaluation discrepancy.

### Primary references

- Haarnoja SAC / Spinning Up: https://spinningup.openai.com/en/latest/algorithms/sac.html
- Delalleau et al. Hybrid SAC: https://arxiv.org/abs/1912.11077
- Fu et al. Hybrid-MASAC (CoRL 2025): https://proceedings.mlr.press/v270/fu25c.html
- Yu et al. MARL benchmark workshop: https://nicsefc.ee.tsinghua.edu.cn/%2Fnics_file%2Fpdf%2F65ee92d0-74dd-4bfb-b7d4-46cec7b3385d.pdf
- Archived user W&B study:
  https://wandb.ai/truongdaoanhduy-uit/uav_search_target
