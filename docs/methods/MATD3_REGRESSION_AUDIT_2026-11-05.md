# MATD3 seed-44 regression audit — 2026-11-05

Observed on 2026-11-05, Asia/Ho_Chi_Minh. Scope: compare the two explicitly identified 2026-11-02 runs with the two 2026-11-05 runs; inspect W&B history/logs, resolved config, exact Git sources, and model weights. No algorithm changes or production training were performed.

## Verified result

All four runs finished 20,000 episodes with stop_reason=episode_budget.

| Run | W&B ID | Joint transitions | Final rolling return | Final rolling search | Final rolling delivery |
|---|---|---:|---:|---:|---:|
| 02 Nov BASE, suffix e3778d | db61bd0c | 24,219,648 | -160.0645 | 60.75% | 59.00% |
| 02 Nov PQL, suffix c338a0 | d51b0350 | 24,813,568 | -34.4865 | 69.75% | 67.75% |
| 05 Nov BASE, suffix 5796b3 | 140e6169 | 30,720,000 | -4221.4810 | 0% | 0% |
| 05 Nov PQL, suffix 2ed224 | ead641c4 | 30,720,000 | -4221.4810 | 0% | 0% |

The rolling window is 20 completed episodes in every run. These are not best-ever or single-episode returns. Whole-training mean delivery: old BASE 48.296%, old PQL 49.58775%, new BASE/PQL 1.596%. All have zero final rolling mission success; the better older result is not evidence of solving all 20 targets.

Links use https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/<ID>.

## Directly reproduced failure mechanism

Downloaded both old final policies from their W&B checkpoint manifests, SHA-256 verified:
- db61bd0c: 4a62da47403e10f188db3a074d80aadedf76ce6e3635ce8b4c7ee80773c23eec
- d51b0350: 55897da8856e146b15bde8097767dde2f3bf61fa41bb13aa8aa3463003a0bcae

Downloaded both new validation-best actor snapshots. They were selected at global_step=3,686,400 and policy_version=3,476 completed actor updates. Snapshot episode=0 means no vector environment had yet finished its first 1,500-step episode, not an untrained actor.

On actual reset observations for seeds 44, 1000000044 and 2000000044, new XYZ pre-tanh logits are approximately [+20, -15, -19]. All six actors output [+1,-1,-1]. CPU FP32 autograd measured exactly zero tanh derivatives for every XYZ dimension at these resets. Input ranges were approximately [-0.22,1]; no giant unnormalized input was present in these observations.

A local counterfactual kept the trained weights and changed only the pre-tanh normalization calculation: its gradient became nonzero, max about 0.029. This is a derivative probe, not retraining evidence or a demonstrated convergence fix.

A bounded deterministic rollout using common unchanged environment code, seed 1000000044, reproduced:
- New BASE/PQL: all six UAVs at y=0,z=0 by step 30. At step 300, search/delivery remain zero; aircraft are clustered along the southeast boundary.
- Old BASE final checkpoint: delivery 20% by step 300, multiple flight altitudes and locations.
- Old PQL final checkpoint: delivery 10% by step 300, multiple flight altitudes and locations.

These old-final/new-best probes are for reproducing the mechanism, not a matched-training-budget benchmark. Inference ran on CPU, PyTorch 2.7.1+cu126. The numeric results and states are retained in the linked evidence JSON.

The camera footprint radius is max(0,z)*tan(FOV/2), so ground-level flight has zero footprint. Lack of discoveries also removes report-bearing communication opportunities; destination masks generally allow only silent and canonical_continuous gates RF power to zero. Thus the remaining hybrid branches do not reliably supply a way out once the motion head is saturated.

W&B gradient history sampled at 3,000 points per run shows new actor norm about 1e-9 near six million transitions, and predominantly zero thereafter; sampled first zero is at 7,274,496 transitions, not necessarily the first zero in full history. Occasional tiny nonzero total actor norms can arise in other branches. Old final actor norms remain positive (~0.074/0.070); new final norms are zero. Norm definitions changed (global vs maximum tensor), so magnitudes are not directly comparable across modes, but zero/nonzero remains meaningful.

Final 20-episode means:
- New shield interventions 99.5017%, commanded stalls 97.6472%, movement saturation after noise 53.9830%.
- Old BASE: 55.0852%, 3.8192%, 8.2690%.
- Old PQL: 52.3588%, 3.5214%, 8.2213%.

The movement saturation statistic measures executed actions after Gaussian exploration noise/clamping. It does not contradict 100% saturation of the deterministic motion head at the tested reset observations.

In the new final episode, blocked motion, unsafe actions, shield interventions and correction sum to -4213.8181, 99.2686% of return -4244.8667. Final battery fraction is ~22.18%. These data support an unsafe stuck policy. There is no evidence here that CBF code or penalty weights changed and caused the regression.

## Exact source and config comparison

Old source: 05beef6cd0d97d39d51b220f9f7789e915b5bc10.
New BASE: 574f36c6d36d6182d3204a53f9e497498d3470f8.
New PQL: ab2e953d37520eb5896700f1264d1e4ba80fdfad.
Current inspected checkout: da3033d, experiment/hfbest-embedded-source-steps1500-20261105.

No algorithms/environment/communication differences between the two new source commits. No environment/communication source diff between old and new BASE. The resolved world, sensor, network, energy and reward namespaces match exactly.

| Difference | Old | New |
|---|---|---|
| MATD3 output normalization | true | false |
| Gradient clipping | global norm 10 | per-parameter norm 0.5 |
| Behavior discrete exploration | epsilon-greedy, 1→0.05 over 14,745,600 transitions | categorical/Gumbel; epsilon fields are unused |
| Other-agent actions in actor loss | detached current policies | replay minibatch |
| Actor/target categorical representation | hard one-hot / straight-through | soft Gumbel, while replay stays hard |
| compile_models | true | false |
| vectorize_critics | true | false |
| critic_chunk_size | 1024 | 0 |
| collector_overlap config | true | false |
| PyTorch build on Kaggle | 2.11.0+cu128 | 2.11.0+cu130 |
| Python on Kaggle | 3.13.15 | 3.13.16 |

Both use T4x2, AMP off, fused optimizers off, seed44, 6 UAVs/20 targets/15 obstacles/1000m/1500 horizon, LR3e-4, batch8192, replay1M, num_envs4096, eight updates per vector step, and shared actor/critic. PQL reports scheduling_deterministic=1 for both cohorts.

The c0a935e source-backed-baseline commit groups multiple algorithm changes. W&B already shows degraded runs from Nov 3. The observed direct cause is saturated/dead motion gradients; disabling output normalization is the strongest candidate because it removed the old bound on pre-tanh XYZ and the same-weight derivative probe isolates that calculation. Attribution of the complete training regression to that single flag is not proven: sampler, peer actions, clipping, execution and software also changed.

## Baseline source audit

Read primary source:
- https://github.com/sfujim/TD3/blob/master/TD3.py
- https://github.com/JohannesAck/MATD3implementation/blob/fd6c7d0df4fc4effd4e0fa11abdbc7d12d8f01a1/matd3/matd3/trainer/matd3.py
- The same pinned source's common/distributions.py and common/tf_util.py.
- Stable-Baselines3 TD3 official source and docs via Context7.
- docs/methods/REFERENCE_BASELINE_PROVENANCE.md.

Twin target minimum, policy delay and Gaussian smoothing match the inspected TD3 core. Replay peers and per-variable 0.5 clipping have author-source support. Original TD3 has no output normalization, so reinstating it must be a named ablation/variant, not silently claimed as vanilla.

Current implementation is not a full reproduction of the Ackermann source:
- Shared actor/critic are project choices; author networks are independent.
- UAV hybrid action, masks, RF argmax gating, privileged critic state and hard-replay/soft-training representation are task adapters.
- Ackermann actor source includes 1e-3 * mean(square(flatparam)); the current actor objective deliberately excludes that regularizer under the recorded paper-equation preference.
- SB3 TD3 supports continuous Box actions; it is not a drop-in implementation of this hybrid UAV action space.

The canonical YAML contract test uses independent parameters. A shared override is not a literal reproduction of that profile. Existing tests establish their checked invariants, not convergence.

## Next controlled experiment, not yet performed

1. Pin one upstream implementation/commit, document shared/hybrid adapters and the actor-regularizer decision, and pin PyTorch/CUDA.
2. Single-seed44 A/B: current configuration vs only output_normalization=true as a clearly named variant. Compare raw XYZ, tanh derivative, altitude, stall rate and held-out delivery at matched transitions.
3. If the baseline must retain output_normalization=false, test learning-rate choices under a fixed protocol (e.g. 3e-4 vs 1e-4); this is a candidate, not an established fix. Isolate clipping and hybrid representation/sampler changes separately.
4. Log raw motion head magnitude, XYZ Jacobian and z=0 occupancy. Detect persistent dead motion gradients before spending the full 20k budget. Preserve safety/reward during the first ablation.

## Verification

Executed on current checkout:
python -m pytest -q tests/test_reference_baseline_contract.py tests/test_reference_critic_gradients.py tests/test_matd3.py

Result: 24 passed in 3.12s.
No learning-performance fix or 20k diagnostic retraining is claimed.

Evidence: ../superpowers/evidence/2026-11-05-matd3-saturation-audit.json.
Original raw API downloads and diagnostic harnesses are under /tmp/matd3-audit-20261105 (transient).
