# Online training diagnostics

Purpose: use actual rollout, replay-batch and fixed-seed evaluation evidence to
investigate weak learning. Warnings identify symptoms; they do not prove a
particular code bug or guarantee discovery of every logical error.

## Metric interpretation

| Evidence | Metrics | Interpretation |
|---|---|---|
| Mission outcomes | episode/return, search/delivery/success | Higher return alone can hide poor delivery. Compare all three outcomes. |
| Recent trend | rolling/return_mean/std/change, rolling/delivery_rate_percent | Non-overlapping window comparison; warm-start windows use new episodes only. |
| Generalization | eval/mean_return, search/delivery/success | Deterministic CPU evaluation with the same held-out seeds every interval. |
| Reward accounting | reward/discovery/delivery/blocked_motion/expired_report/dropped_report/propulsion_energy/communication_energy | Cumulative components should sum to the actual episode return. |
| Critic | learner/q_mean/q_abs_max/target_mean/target_abs_max/td_error_abs_mean/max/twin_q_gap | Large Q/TD error or critic disagreement is evidence to inspect, not proof of divergence. |
| Optimizer | learner/actor_grad_norm/critic_grad_norm/alpha_grad_norm/critic_gradient_clipped | Norm is measured before clipping. Finite-gradient checks remain enforced every update. |
| MASAC exploration | learner/continuous_entropy/discrete_entropy/alpha_continuous/alpha_discrete | Watch entropy together with return, destination availability and temperature. |
| MATD3 | learner/policy_updated/actor_update_step | Actor metrics refer to the last actual delayed actor update, never an invented zero loss. |
| Actions | action/silent_fraction/peer_fraction/gcs_fraction/movement_saturation_fraction | Distinguish no transmission, relaying, direct GCS and saturated movement. |
| Reports | reports/buffer_count/buffer_bytes/expired/dropped | Search high with delivery low can involve queues, expiry, routing or policy choices. |
| Network | packet/PHY delivery, delay, throughput, hops, collisions | Packet delivery and complete target-report delivery are different metrics. |
| Safety/energy | blocked/boundary rates, propulsion/RF energy, exhausted UAVs | Explain penalties and early battery termination. |
| Run/runtime | account, job/run UTC timestamps, source SHA, GPU mask, elapsed hours, TPS/replay/updates | Reproduce the experiment and identify compute bottlenecks. |

Defaults: rolling window20 episodes, delivery gap40 percentage points,
return drop20 reward units, blocked-motion rate0.3. Thresholds are configurable
under tracking, and are heuristics for this scenario. Warnings require a full
window except reward-accounting checks; repeated events have a window cooldown.
No-search, report-loss and battery-exhaustion warnings are observations of
mission failure modes, not claims that they necessarily indicate faulty code.

## Files and console

The learner owns the online W&B run and captures stdout/stderr with console wrap.
Structured warnings appear in Logs and diagnostics.jsonl in Files.
Startup failures before a W&B run exists retain local/Kaggle evidence.
Fatal failures write failure.json with a redacted traceback, algorithm, seed,
global step, episode counter, update counter and last learner metrics.
An optimizer/runtime failure also saves failed_batch.pt containing the actual
sampled replay tensors, and failure_state.pt preserves model tensors after the
failure, with nonfinite parameter names in the context. Some optimizer steps may
already have completed; this is diagnostic evidence, not an exact restart snapshot.
These are uploaded before finishing the failed run.
Resolved config, episode histories at checkpoints, CPU evaluation metrics and
evaluation logs are retained. Credentials and generated outputs are Gitignored.
A hard process/machine kill can prevent the final failure upload; periodic online
checkpoints and prior console/diagnostic files are the durable evidence.

Fixed-seed evaluation runs in a separate CPU process, preserving learner RNG.
Default interval50 episodes, timeout300 seconds, runtime.eval_episodes=3,
runtime.eval_seed=10044. Set tracking.evaluation_interval=0 to disable it.
Evaluation failure is logged separately and does not silently become a good score.
Evaluation consumes CPU time and is included in the job's wall-time budget.

## Reading the dashboard

Start with fixed-seed delivery/success and rolling return. Then use reward,
actions, network and energy to explain changes. For numerical failures, inspect
Q/target/TD/gradient/entropy and failure.json + failed_batch.pt.
Use the run's System tab for native CPU/GPU/VRAM telemetry and Logs for output.
The project table includes Kaggle account, start UTC, GPU mask and elapsed hours.

Jobs already running are pinned to their submitted SHA. Updating local/GitHub or
the dashboard cannot inject new metrics into their processes. Existing plots keep
updating; enhanced logging starts in jobs launched from the new source.

## Sources

- OpenAI Spinning Up SAC logs Q values, policy/critic losses and log-probabilities:
  https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/sac/sac.html
- SAC deterministic evaluation:
  https://spinningup.openai.com/en/latest/user/saving_and_loading.html
- W&B custom axes, scalar/media logging:
  https://docs.wandb.ai/models/track/log
- W&B console logging:
  https://docs.wandb.ai/models/app/console-logs

Additional reward/action/window checks above are project-specific diagnostics,
not standardized guarantees of convergence.

## Compact display profiles

The main saved view now has18 panels and an exact filter for the two selected
long-training run IDs. Other experiments, validation/recovery and superseded runs
remain stored but do not clutter this view. The Runs table keeps run name (including algorithm), state, Kaggle account,
start UTC, runtime, completed episodes, return, search and delivery.

Deep diagnostics use a separate20-panel profile. Logging and failure files remain
available for investigation; changing display profiles does not modify learners.

```bash
python -m uav_network dashboard --config configs/kaggle.yaml   --profile core --run-ids 9c9309b3 afdad4a9   --workspace-url 'https://wandb.ai/truongdaoanhduy-uit/uav_search_target?nw=gmcixx5iveu'
# Use --profile diagnostics in a separate workspace when investigating an issue.
```

Removed from the main display: duplicate rolling/live outcome charts, optimizer
counts/replay/timing charts, separate propulsion/RF charts, extra packet/PHY charts,
individual warning flags, Q/gradient/entropy/alpha and detailed action/reward plots.
These remain logged or are available in the diagnostic view and run Files.


## Explicit Runs-column visibility

Pinning columns alone did not hide unset columns in the W&B frontend.
dashboard_columns.py now enumerates existing project config/summary keys and
saves explicit false visibility for everything outside the nine-column allowlist.
Run metadata/config/history remain intact; only table presentation changes.

Verified by reading the saved backend view on2026-10-13:
9 visible columns,933 explicitly hidden, including244 legacy
training_handoff.resolved_config.* columns. Keep: Name, State, Created,
Runtime, Kaggle account, completed episodes, episode return, search %, delivery %.
Hide: old experiment/postprocess/Hugging Face/secret-name/visualization/W&B
configuration, repeated hyperparameters, internal counters and extra diagnostics.
Those details remain accessible in individual runs, config files and logs.
