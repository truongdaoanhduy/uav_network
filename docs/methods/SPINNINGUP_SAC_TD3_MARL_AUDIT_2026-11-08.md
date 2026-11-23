# SAC / TD3 → UAV MASAC / MATD3: paper-alignment audit (2026-11-08)

## Scope and baseline identity

- Audited working branch: `experiment/hfbest-embedded-source-steps1500-20261105`, starting at `cc1885f`.
- Learners: `uav_network/algorithms/masac.py`, `matd3.py`, `models.py`, `action_utils.py`.
- Interfaces: `uav_network/experiments/training.py`, `algorithms/replay.py`,
  `environment/env.py`, `environment/energy.py`, `environment/xyz_motion.py`.
- No change to source math, policy outputs, replay schema, reward, task physics, safety,
  sensor, actor observation, critic global state, learned RF power, checkpoint, PQL,
  U8/U16, seed 44, or existing exported Kaggle notebooks.
- This audit is **not** a 20k-episode GPU learning benchmark. Unit correctness
  is not evidence of sample efficiency or asymptotic performance.

## Primary algorithm and released-code references

1. [OpenAI Spinning Up SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html):
   continuous-action entropy-regularized off-policy actor-critic; squashed
   Gaussian stochastic actor, twin target Q and soft Bellman target.
   The linked implementation does not provide the UAV discrete action branch.
2. [OpenAI Spinning Up TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html):
   deterministic actor, clipped double-Q target, target policy smoothing,
   delayed actor and target updates.
3. [Haarnoja et al., SAC algorithms and applications](https://arxiv.org/abs/1812.05905):
   automatic entropy-temperature tuning is a documented SAC variant.
4. [Ackermann et al. 2019, MATD3](https://arxiv.org/abs/1910.01465),
   Section 4 Eq. (5): independent per-agent centralized double critics and
   Gaussian target-policy smoothing for **all** agents' target actions.
5. [Ackermann original MATD3 source](https://github.com/JohannesAck/MATD3implementation)
   and [authors' TF2 multiagent repo](https://github.com/JohannesAck/tf2multiagentrl):
   peer replay actions in actor optimization, differentiable categorical policy.
   The TF2 per-agent learner adds target smoothing to the agent being updated;
   this **implementation detail is narrower than the published Eq. (5)**.
   The paper is the primary reference for the baseline.
6. [Delalleau et al. 2019, hybrid continuous/discrete action SAC](https://arxiv.org/abs/1912.11077):
   hybrid-action interface and finite categorical expectation; not a mandate
   to replace the SAC actor/critic with another algorithm.

**Terminology:** No unique, universal "original MASAC" source exists across
centralized-critic, factorized-value, attention-critic and hybrid MASAC works.
This repo's approach should be described as **CTDE Hybrid-MASAC based on SAC**,
and **CTDE Hybrid-MATD3 based on Ackermann MATD3**, with documented UAV adapters.
It is incorrect to claim byte-for-byte reproduction of Spinning Up.

## Executed code-level findings

| Component | Status | Evidence and action |
|---|---|---|
| MASAC reparameterized Gaussian, tanh-Jacobian, twin Q | CORE VERIFIED | `models.py` and `masac.py`; unchanged |
| MASAC critic soft target and entropy actor loss | CORE VERIFIED | `update_critic`, `update_actor` and existing mathematical regression tests; unchanged |
| MASAC Shared team Q expectation | VERIFIED UNDER FACTORIZED JOINT POLICY | Each agent's discrete alternative is exactly enumerated while peers are sampled. Mean of per-agent conditional Q expectations is an unbiased joint-Q estimator if per-UAV action samples are independent. New exhaustive non-additive two-agent Q test passes. Added explanation comment only. |
| SAC two per-agent auto temperatures | DECLARED HYBRID EXTENSION | No unapproved temperature or mask change |
| MATD3 twin per-agent centralized critics, delay, target Polyak | CORE VERIFIED | `matd3.py`, existing numerical/source tests; unchanged |
| MATD3 Gaussian smoothing on **all** UAVs | PAPER VERIFIED | Ackermann Section 4 Eq. (5) explicitly uses all agents. New tests for Shared and Independent both pass. Added a clarifying docstring only. |
| MATD3 actor gradient through own action, replay peers | CORE VERIFIED | `update_actor` and existing `test_reference_baseline_contract.py`; unchanged |
| MATD3 soft Gumbel actor/target vs hard one-hot replay | **KNOWN REPRESENTATION DIFFERENCE**, not proven bug | A nonlinear toy-Q test exhibits a difference. Do **not** silently switch to hard=True or change replay policy: would introduce a new estimator/adapter requiring comparative validation. |
| CBF action vs replay | VERIFIED COMMAND SEMANTICS | Environment applies CBF inside motion dynamics; replay stores the **issued** control and destination. CPU integration test forces boundary intervention and verifies that exact issued action is retained. This is valid when the action in the MDP is defined pre-shield; it does not require changing replay to post-shield velocities. |
| Actor local observations / critic global state | ARCHITECTURE VERIFIED | No actor access to `next_state` or joint critic state through neural input. Runtime safety-module decentralization is a separate system claim. |

## New executable regression tests

File: `tests/test_paper_faithfulness_audit_20261108.py`.

1. Exhaustive 2-agent, 3-destination, *non-additive* Q matrix:
   MASAC shared conditional-enumeration expectation equals exact
   `sum_d1 sum_d2 pi1(d1) pi2(d2) Q(d1,d2)`.
2. MATD3 independent: with categorical sample fixed, changing Gaussian target
   noise changes both UAVs' continuous next actions, not their discrete branch.
3. Same invariant for MATD3 shared parameters.
4. Soft/hard categorical critic-input distinction is demonstrated with a
   nonlinear synthetic Q and valid finite gradients; the test makes no claim
   about the size of the error on a trained critic.
5. A CPU CBF boundary case stores pre-shield control in replay, while the
   environment prevents the out-of-bounds motion.

These tests complement existing `test_algorithm_math_regressions.py`,
`test_reference_baseline_contract.py`, `test_reference_critic_gradients.py`,
`test_parameter_sharing.py`, `test_matd3.py`, `test_masac.py` and
`test_split_updates.py`.

## Risks not resolved by unit testing

- MATD3 soft categorical critic targets and hard categorical replay have
  different supports. A numerical toy example confirms the possibility of
  Q discrepancies, **not** the size or impact on a trained checkpoint. A
  fixed-checkpoint same-replay Q diagnostic would be the next research step.
- Shared-team-critic MASAC's unbiasedness does not make its gradient variance
  identical to a fully enumerated joint policy objective. Its Monte Carlo
  sampling remains a deliberate computational approximation.
- Actual MATD3 and MASAC training curves, validation return, reproducibility,
  CBF intervention rate, wall-clock time and memory require run-level checks.
- These tests do not assert that a particular paper's attention critic or
  value-decomposition mixer was replicated; those are different algorithms.

## Change decision

**Do not replace the original SAC/TD3 loss definitions, change the Gumbel
estimator, or alter the environment action semantics without failing
reference-equation tests or controlled evidence.** The actual code changes
from this audit are source-clarifying comments/docstrings and a new set of
regression tests plus this provenance report. No unverified algorithm
"fix" was introduced.
