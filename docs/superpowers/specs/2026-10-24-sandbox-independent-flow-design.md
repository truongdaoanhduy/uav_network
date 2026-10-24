# Sandbox-independent Kaggle training flow

## Intent and authorization

The user explicitly requires that after seeing a W&B run running, they can turn off Sandbox/their computer while training continues across sessions and eventually creates a CPU visualization session. Keep the existing models, scenario, budgets, same accounts, and healthy jobs. Implement within the existing Kaggle/private-dataset pipeline. The choices below are implementation judgments under that request, not separately chosen user preferences.

## Observed gap

GPU notebooks already submit their next GPU segment or final CPU visualization. Insufficient quota is recorded as waiting_for_quota, but nothing on Kaggle wakes that lineage later. The external watcher was paused at02:19 Oct24 after Sandbox Resource not found. Local-host availability therefore cannot be the primary recovery mechanism. Current three healthy1Mchildren use337608c2, which includes staged-input cleanup; recovered MASACPQL1M and BASE500k c02 use tested4cd9358. Preserve them.

## Approaches

- GPU notebook self-chain alone: existing happy path works, but a quota wait survives only as a file and has no future executor.
- ChatGPT/local Sandbox watcher: useful secondary monitoring, fails the explicit requirement when the local host is offline.
- **Chosen implementation:** one CPU-only Kaggle controller per lineage, using existing private same-account launch credentials. It follows normal GPU auto-continuations, resumes quota-paused terminal parents when quota permits, adopts final CPU visualization, and chains its own bounded CPU sessions when needed. This reuses the current submission and completion validators.

## Goals and constraints

- No runtime read of the local key.txt, local filesystem, local daemon or ChatGPT automation is required after deployment.
- Keep report1MB,35targets,25obstacles,4096worlds,batch8192,1800horizon,target20000,seed44; preserve all science/RNG/state.
- Same account/lineage only. Observe/enforce current GPU quota. Wait through lack of quota; never rotate accounts or bypass limits.
- Do not cancel, restart, overwrite, or inject code into healthy existing GPU jobs.
- Completion means verified episode budget plus final CPU visualization/publication; a Finished segment or benchmark is not full training.
- No secret value in source, job specs, logs, W&B config or public Git history. Use private Kaggle datasets already used by this project.
- Preserve failed-attempt evidence. Existing failed children are adopted/inspected, never overwritten automatically.

## Controller behavior

1. Bootstrap a private CPU-only Kaggle kernel from an immutable source commit; mount the clean W&B dataset and same-account pipeline token. Set CUDA_VISIBLE_DEVICES empty and route to a new cloud.controller module, not cloud.remote's CUDA runner.
2. Maintain one atomic controller-state.json plus W&B heartbeat: root/current GPU reference, phase, verified prefix, current CPU visualization reference, next controller reference, and last error category. Do not store secrets.
3. While a GPU parent is running/pending/queued, observe only. On a terminal parent, download small control JSON files and validate account/ref/results with existing next_stage.
4. If the parent already submitted a descendant, verify actual owned existence and adopt it. Any existing child, including failed or status-ambiguous private children, must not be overwritten.
5. A valid time-budget segment without a child waits for sufficient same-account quota, then uses existing submit_continuation from an explicitly pinned tested GPU source; reset an old quota-shortened timeout against current verified quota, capped39600s. Exact input prefix/world-state checks remain in the GPU bootstrap.
6. A verified complete episode budget adopts an existing CPU visualization or invokes existing submit_visualization. Wait for that CPU job; require completed result/media publication evidence before marking the whole pipeline complete.
7. Unknown failures or unusable full-state summaries become explicit blocked status; do not manufacture counters, reuse policy-only state as exact resume, or blindly spend quota retrying a deterministic failure.
8. Before the controller's own CPU session deadline, persist the state in a uniquely named next CPU controller spec, submit it on the same account, and exit after confirming/adopting that successor. A stable root hash plus generation prevents collisions; an existing successor is adopted, not overwritten. This is normal bounded-session chaining, not an interactive keepalive.

## Integration and ownership

- Existing GPU auto-continuation remains the happy-path owner. The controller acts only after a GPU parent is terminal, when its own automatic submission is complete; it reconciles actual owned descendants before any mutation.
- New CLI --auto-continue launches must seed a CPU controller and record its reference/readiness. A controller launch failure preserves the already-running GPU manifest and reports that unattended readiness was not established.
- Seed controllers for the five existing lineages after tests. Existing GPU jobs keep their source and state. Future recovery GPU source is explicitly4cd9358; merely running a newer controller does not upgrade a child's source.
- Retain the ChatGPT watcher only as secondary observation; it must not compete with an active cloud controller.

## Verification

- Test generated CPU bootstrap execution/metadata, private credential mounting, no GPU request and no local-key dependency.
- Test active-parent observation, quota exhaustion then replenishment, immutable completion gates, child adoption/idempotency, refusal to overwrite failed children, CPU completion/publication checks, controller-session handoff, and secret rejection.
- Run full CPU suite and independent review. Existing Task7 two-T4 smoke remains separately required.
- Run a separately labelled tiny cloud end-to-end smoke with real W&B/HF publication: mini GPU training -> CPU visualization, supervised by a CPU controller, with no intervening Sandbox-driven submission. Production20k budgets remain unchanged.
- Verify all five production controllers are running/heartbeating in W&B and carry the actual current lineage before saying Sandbox can be closed.

## Limits

Kaggle quotas, service outages and credential revocation still constrain execution. The controller waits for legitimate quota and makes unrecoverable failures visible. It does not promise20k completion within12h or silently change the scenario. Model-quality comparisons require full-budget results, not this workflow smoke.
