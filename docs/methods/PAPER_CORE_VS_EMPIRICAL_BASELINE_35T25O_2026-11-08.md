# Paper-core vs empirically selected MASAC/MATD3 baseline — 35T/25O, 2026-11-08

## Decision: distinguish the original equation structure from the best measured architecture

**The existing Shared H512 U8 ActorLayerNorm MATD3 run is a successful
*parameter-shared CTDE variant*, not the unmodified per-agent topology in
Ackermann et al. (2019).** The existing Shared H256 U8 ActorLayerNorm MASAC is
likewise a valid hybrid CTDE variant, not a byte-for-byte reproduction of
Fu et al. (2025), BenchMARL MASAC, or single-agent SAC.

**Use two unambiguous labels**, neither silently replaces the other:

1. `PAPER_CORE_CTDE_HYBRID` — profile
   [configs/comparison-paper-core-35t25o-xyz-sensing.yaml](../../configs/comparison-paper-core-35t25o-xyz-sensing.yaml).
   Six *independent* local actors; six *independent* centralized twin critics
   (each module contains Q1 and Q2); **no Actor LayerNorm**; no PQL;
   MATD3 peer actions drawn from replay during actor update, and target
   policy Gaussian smoothing; MASAC reparameterized tanh Gaussian,
   exact finite discrete expectation, double Q and two temperatures.
   **Zero existing W&B runs exactly match Independent+NoLN+H512+U8** among
   the archived ten. This *implementation profile is now contract-checked,
   not a completed experimental result*.
2. `EMPIRICAL_SHARED_ACTOR_LN` — the existing ten-run experimental family:
   MATD3 Shared H512 U8 ActorLN `ccaadcd6`, best validated delivery
   58.86% (20k complete); compute-efficient MATD3 Shared H256 U8
   ActorLN `635d7134`, best 51.43% (20k); MASAC Shared H256 U8
   ActorLN `7f5fd73f`, best 52.00% (20k).
   See [full 10-run matched-budget analysis](WANDB_10_LATEST_35T25O_COMPARISON_2026-11-08.md).
   These are *single-seed* findings with 5-episode periodic validation;
   the best percentages are at different checkpoint budgets.

The independent vs shared MATD3 match at H512 U16 ActorLN favors shared
by +42.86 delivery percentage points at an identical checkpoint, so
sharing is justified as a **proven good choice for this scenario**.
It does **not** isolate Actor sharing vs Critic sharing. There is no
Shared NoLayerNorm paired ablation, therefore **Actor LayerNorm remains
an empirical architecture choice rather than a paper-faithfulness claim**.

## Source-equation audit at code-line level

### MATD3: original continuous mathematical core

Primary original source: Ackermann et al. 2019, v2
[arXiv 1910.01465](https://arxiv.org/abs/1910.01465),
Section 4, Eq. (5); author
[source repository](https://github.com/JohannesAck/MATD3implementation).
TD3 background: Fujimoto et al. 2018 and
[Spinning Up TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html).

- Centralized twin critic per agent:
  `MATD3.__init__`, `uav_network/algorithms/matd3.py:45-50`,
  `TwinCritic`, `uav_network/algorithms/models.py:77-86`.
  Per-agent modules exist if `share_critic_parameters=false`.
  Shared mode has **one team twin critic instead** and must be labeled.
- Clipped-double Bellman target:

  `y_i = r_team + gamma * (1 - terminated) * min(Q1'_i(s', a'), Q2'_i(s', a'))`.

  Implemented by `td3_target` in `matd3.py:19-20`, invoked in
  `MATD3.update_critic` at `matd3.py:215-241`.
  One cooperative team reward supplies `r_i=r_team` for each UAV.
  MSE on both Qs is computed against the *same* target.
- Target-policy smoothing: `target_actions` at `matd3.py:128-151`;
  independent Gaussian noise on **every UAV continuous target vector**
  with sigma 0.2, clipping at ±0.5, and continuous action clamp to [-1,1].
  The paper's Eq. (5) adds target noise to each agent. Noise is **not**
  applied to invalid/discrete one-hot routing decisions.
  This last restriction is a hybrid-action UAV-specific adaptation.
- Delayed updates and Polyak: `matd3.py:304-320`.
  Critic updates on every learner step, Actor plus targets once every
  `policy_delay=2` Critic steps; target averaging `tau=0.005`.
  Tau is **our chosen experimental hyperparameter**, not an assertion
  that all upstream MATD3 experiments used precisely 0.005.
- Actor objective (per agent, per-agent critics):

  `L_actor_i = - E_s Q1_i(s, a_1^replay,..., pi_i(o_i),...,a_N^replay)`.

  In `matd3.py:243-302` peer actions are detached from replay,
  only agent `i`'s candidate is replaced. Actor is decentralized:
  local observation only. Critic 1 is used for the actor gradient.
- Author source (not the original TD3 paper itself) clips each tensor
  gradient norm to 0.5. Our per-variable mode matches its gradient
  convention; `output_normalization=false`,
  `initialization=default`, no LayerNorm in paper-core.

**Cannot claim byte-for-byte author MATD3:** The user's UAV environment
requires one 4-D continuous vector (XYZ acceleration and RF power)
and a categorical, masked destination (7 possible actions including
silent). The upstream MPE model is not this same action space.
The UAV code uses `F.gumbel_softmax(..., tau=1, hard=False)` for
differentiable discrete critic input, while the **actual transition
and replay** use an integer destination/hard one-hot.
The published MATD3 actor/critic equations do not specify this
exact UAV routing adapter. The upstream MATD3 code also has a
differently shaped communication action distribution, so no exact
author-code replay equivalence should be claimed.

### MASAC: source-backed hybrid CTDE mathematical core

There is no **single universal original MASAC** implementation.
Cite the selected combination precisely:
[Haarnoja SAC](https://arxiv.org/abs/1812.05905),
[Spinning Up SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html),
[Delalleau Hybrid SAC](https://arxiv.org/abs/1912.11077),
[Fu et al. Hybrid-MASAC](https://proceedings.mlr.press/v270/fu25c.html),
and [BenchMARL MASAC](https://github.com/facebookresearch/BenchMARL).
Do **not** claim replication of Fu's attention critic network.

- Local actor: `HybridActor` in
  `uav_network/algorithms/models.py:30-50`;
  tanh-reparameterized Gaussian continuous action,
  log-std bound `[-20,2]`, stable tanh log-Jacobian,
  masked categorical policy for destination.
- Each agent's independent twin centralized critics `Q_i1,Q_i2`:
  `MASAC.__init__` at `masac.py:25-57`, `TwinCritic` at
  `models.py:77-86`.
- Entropy-regularized target in `masac.py:206-249`:

  `V_i(s') = E_{a_i,d_i, a_-i'}[ min(Q_i1',Q_i2')
                               - alpha_i,c log pi_i,c
                               - alpha_i,d log pi_i,d ]`
  `y_i = r_team + gamma * (1-terminated) V_i(s')`.

  Continuous and peer actions are sampled; the small finite
  categorical own-destination space is enumerated, yielding an exact
  *conditional* categorical expectation rather than sampling it.
  This is a multi-pass Hybrid SAC architecture, not Fu's exact attention
  model.
- Actor in `masac.py:259-324`:

  `L_actor_i = E[alpha_i,c log pi_i,c
                 + sum_d pi_i(d|o_i)
                         (alpha_i,d log pi_i(d|o_i) - minQ_i(s,a_-i,a_i,d))]`.

  Differentiable Gaussian reparameterization and exact masked
  destination expectation; other UAVs' actions are detached.
  SAC Actor updates every optimizer iteration, unlike TD3 delay.
- Per-agent continuous/discrete alpha in `masac.py:325-352`.
  `temperature_loss=alpha` uses the original SAC
  `alpha=exp(log_alpha)` dual parameter gradient. No hard alpha
  clipping, no target entropy annealing.
- **Mask-aware target is a declared, reasonable UAV-specific extension.**
  A destination mask with only one valid action (silent) has
  categorical entropy zero, so a fixed positive discrete entropy
  target `0.2*log(7)` would be unattainable at that state.
  The paper-core profile uses `0.2*log(K_valid)` to define
  an attainable target under masked communication action support.
  This should be reported explicitly, not misrepresented as an
  unmodified upstream BenchMARL setting.

### Non-algorithmic adapters that are still present

- A centralized critic receives full **privileged global state**
  (830-D for 6UAV,35T25O) and the joint action, while each Actor
  receives only its **228-D local observation**.
  This is CTDE, not centralized execution; it may expose true scene
  geometry to the Critic but **not** to the Actor.
- CBF safety intervention and commanded-action replay define the
  environment transition. Rewards, sensor belief, routing,
  link budget, energy and generated world remain unchanged.
- Inactive UAV motion and silent RF power are deterministically gated;
  masked invalid destinations receive zero probability.
- Hyperparameters H512x512, batch8192, U8 per vector collection,
  4096 envs, replay1M, one training seed44, gamma .99, tau .005,
  LR3e-4, 20k episode target are **project choices, not claims
  to match authors' experimental defaults**.

## Source-faithfulness verdict / what must and must not be changed

| Component | Code verdict | Action |
|---|---|---|
| TD3 twin min, replay-Q, delayed policy, target smoothing | Implemented, matches mathematical core | Keep; regression tests |
| MATD3 each agent's actor/Q_i with replay peer action | Implemented in Independent mode | Activate explicit paper-core profile |
| MATD3 masked discrete Gumbel + hard replay | Task adaptation, not exact author replay | Declare; **do not silently change to hard ST** |
| MASAC Gaussian tanh/Jacobian, twin min, Q target | Implemented | Keep; regression tests |
| MASAC finite-action expectation, separate alpha | Source-aligned hybrid formulation, not Fu full architecture | Declare exact sources; keep |
| MASAC masked entropy target | Necessary task-support choice, differs fixed-K BenchMARL | Declare adapter; keep for paper-core |
| Shared Actor/Critic | Parameter-sharing ablation, not original MATD3 topology | **Disable in paper-core; keep empirical runs unchanged** |
| Actor LayerNorm | Not guaranteed by either original SAC/TD3 paper | **Disable in paper-core; keep empirical runs unchanged** |
| Critic LayerNorm | Not part of this baseline | Remains off |
| PQL | Asynchronous training execution variant | Off for direct BASE |
| CBF/reward/network/scene | UAV domain, not SAC/TD3 algorithm components | Keep exactly matched across methods |
| Q(soft) vs Q(hard) on real trained critics | Still unmeasured: best HF snapshots hold only learned Actor weights | Do not invent or claim tested |

**Result of direct code diff:** between the source commit of all
ten historical W&B runs (`a6e07a8f4b199028298b6ffd9a47f2cecd654fc9`)
and the post-audit code, `masac.py` and `matd3.py` changed only in
docstrings/comments — **no mathematical update was silently fixed or altered**.
The source-alignment work here creates a distinct strict profile and an
automated executable contract rather than rewriting already-correct
equations merely for appearance.

## Exact reproducibility and status

Run the **read-only** preflight with no W&B/Kaggle:

```bash
cd /home/aduy/Documents/NCKH/uav_network_study_1200
PYTHONPATH=. python scripts/verify_paper_core_baseline.py
python -m pytest -q tests/test_paper_core_35t25o_baseline.py \
  tests/test_paper_faithfulness_audit_20261108.py \
  tests/test_reference_baseline_contract.py \
  tests/test_reference_critic_gradients.py \
  tests/test_algorithm_math_regressions.py
```

The strict profile is [here](../../configs/comparison-paper-core-35t25o-xyz-sensing.yaml).
Contract evidence is
[here](evidence/paper_core_35t25o_contract_2026-11-08.json).

**Status: architecture and numerical-contract checks completed, no strict
profile full training run performed.** The historical ten results cannot
be relabeled as strict because Independent+NoLN H512 had U16, while
Independent+U8+NoLN is absent. If a paper requires a direct unshared
architecture/source comparison, two **new uniquely labeled strict
experiments** (MATD3 and MASAC at matched environment-step budgets) are
scientifically distinct from the ten existing results; they should be
planned explicitly rather than silently started.
An H256 implementation is not more "original" in core equations than H512;
the width is a controlled hyperparameter.

For reporting strong performance under the actual UAV task,
use the already trained Shared/ActorLN family with the transparent label
`PARAMETER_SHARED_HYBRID_MASAC/MATD3`. This is a legitimate
algorithmic baseline if the study defines parameter sharing a priori,
but it is **not** a strict reproduction of Ackermann's original
per-agent architecture. Do not call one single run statistically superior
without multiple seeds or appropriately described fixed-seed uncertainty.

## References

- Ackermann et al., v2: https://arxiv.org/pdf/1910.01465
- Ackermann released code: https://github.com/JohannesAck/MATD3implementation
- Fujimoto TD3: https://arxiv.org/abs/1802.09477
- Spinning Up TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
- Haarnoja SAC: https://arxiv.org/abs/1812.05905
- Spinning Up SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html
- Delalleau Hybrid SAC: https://arxiv.org/abs/1912.11077
- Fu Hybrid MASAC: https://proceedings.mlr.press/v270/fu25c.html
- BenchMARL MASAC: https://github.com/facebookresearch/BenchMARL
