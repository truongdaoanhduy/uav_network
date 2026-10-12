# Kaggle CUDA and W&B online

User instruction: copy the complete old key.txt to this project, use it to
train on Kaggle with online W&B. Preserve all previous modular/config/file-size
constraints. Local source and destination have identical1461 bytes, mode0600;
key.txt is already Git-ignored. Source remains untouched.

## Design

Use the same JSON schema: kaggle.accounts list with name/token/enabled,
selection policy, wandb api_key/entity/project and unrelated HF fields.
Read only required credentials; never include tokens in generated code,
config, logs, Git, checkpoints or W&B metadata. No dependency on the old repo.

Add a small tracking module to the existing parent training loop. W&B is
explicitly online, uses the key's destination by default, and records progress
before any episode finishes, episode outcomes, safety, report/network metrics,
propulsion/RF energy, learner losses and final aggregates. Final policy artifact
is uploaded. Only the parent process initializes W&B; simulator workers do not.

Keep account/auth/quota commands and kernel generation in uav_network/cloud/.
Use the installed official Kaggle CLI. Read quota before selecting the enabled
account with most remaining GPU hours, resolve its username from owned resources
and validate credential-dataset ownership/privacy before attaching it.
Reuse an existing matching private W&B credential dataset from the old workflow.
Kaggle secrets created through UI are an alternative, not assumed for API pushes.
The complete local key.txt is never uploaded to Kaggle; only the already private
W&B credential source is mounted, and unrelated fields are not propagated.

A generated private script clones an exact published project commit, runs
setup with system packages, confirms CUDA, then runs CUDA regressions and
short online smoke checks for both algorithms. The full job also gates learner
startup on all CUDA regressions and a warmed full-size benchmark of each
algorithm, then runs default1000-episode training for each. On two available GPUs, one independent
learner per GPU; each has2 spawned CPU envs, batch512/replay100000/1 update per
collection (256 sample usages per new transition). On one GPU run sequentially.
Keep real UavNetSim,6 UAVs,10 targets,8 obstacles and800 steps.
Each algorithm has a separate W&B run and output folder. Kaggle output plus
final W&B artifacts persist checkpoints. No speculative4096 SimPy processes.

## Tasks and verification

1. Credential/tracking tests RED→GREEN: legacy JSON, no secret-bearing repr,
   partial progress, complete episode metric mapping, monotone logging, final
   summary and exact exception cleanup. Optional cloud dependencies.
2. Launcher tests RED→GREEN: quota rejection/account resolution, private pinned
   code without keys, explicit GPU placement, command/config overrides and status.
3. Full local suite, independent branch review, publish code; validate the real
   CUDA paths on Kaggle and verify actual W&B history via its API.
4. Fix any remote failure before launching default training. Report kernel/run
   URLs and actual observed state. Do not claim training finished while it runs.

Review focus: expired keys/redaction, secrets accidentally uploaded via code or
artifact, busy/zero-quota accounts, SDK settings inherited by spawned workers,
cloud output-folder collisions and two-device process lifetime/cleanup.

## Progress

Copy complete and byte equality/mode/Git-ignore verified. All7 enabled accounts
have valid quota responses; account_02 currently has30 h remaining. Current W&B
destination: truongdaoanhduy-uit/uav_search_target. Feature branch
`feature/kaggle-wandb` started from315cc932f044db6831e2638d503ab06dc30d5e0b.

Verified `haibro1234/uav-wandb-credential-arch-ablation` is owned by the selected
account and private, and its W&B credential matches the copied local file.
The temporary download was removed. Unrelated HF credentials are not used.

Local tests after launcher regression:69 discovered,66 passed and3 CUDA tests
skipped on this CPU host.
Independent review found two Important cleanup issues, both reproduced before
fixing: tracker setup failure left a W&B run open, and learner termination did
not guarantee simulator worker shutdown. Tracker now finishes failed runs on
partial initialization; SIGTERM executes Python cleanup before a process-group
fallback. A real subprocess test confirms two spawned envs exit with the learner.
The complete suite passed after these fixes.

Real W&B CPU smoke:2 episodes,8 joint transitions,1 update, run
https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/e5c4d730 .
Independent API verification:finished,4 history rows,early progress and1 artifact.
This verifies online tracking, not CUDA or policy convergence. Remote validation
and the default Kaggle training are the remaining execution checks.

First real API push revealed Kaggle generates the new kernel slug from its title,
even when metadata requests an ID. Status on the requested ID was denied; listing
owned kernels located the queued job at its actual title-derived URL. Regression
tests now require title/slug agreement and parsing the canonical push URL. The
launcher also rejects API errors/invalid data sources even if the CLI exits0.
Parent W&B system monitoring is enabled by default, configurable through
`tracking.system_metrics`, to expose CPU/GPU/VRAM usage during real training.
Kaggle capacity is queued, so the full job may be submitted while the validation
job waits. Its own CUDA checks and full-config benchmarks must pass before any
long learner starts; a queue delay does not bypass runtime verification.
