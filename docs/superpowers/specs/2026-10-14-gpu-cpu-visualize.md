# GPU training followed by a separate CPU visualization job

User intent (2026-10-14): keep native 2D UavNetSim and the current independent MASAC/MATD3 models; try longer training; release GPU allocation before doing visualization in a different CPU-only session. The user selected **500 episodes per algorithm**, maximum **1000 steps per episode**.

The existing evaluate command already loads policies onto CPU but is called inside the GPU Kaggle session. Extend that flow with an explicit `--cpu-visualize` option: complete GPU learners and immutable W&B checkpoint uploads, submit one private CPU-only Kaggle kernel containing both successful algorithms, then terminate the GPU job. W&B model artifacts provide checkpoint transfer; a second HF copy is unnecessary.

The long profile inherits the fast execution profile: six UAVs, ten targets, eight obstacles, map1000m, batch512, replay100000, warmup4000, 2 envs per learner, one update/vector collection, independent MLP256x256, seed44/eval10044. Change only max_steps=1000, episodes=500, maximum learner time36000s, output/tag. Existing profiles and default cloud behavior remain available.

CPU visualization runs deterministic native rollouts with the config inside each checkpoint, eval seed10044, one episode. Save PNG, metrics and animation (MP4 if ffmpeg is available, GIF fallback), sampled every5 simulation steps at10fps with the terminal frame retained. State snapshots show paths, FOV, obstacles, battery and target discovery/GCS delivery; no training changes.

The GPU bootstrap may mount a verified owned private dataset containing only that account's Kaggle launch token, required to submit the CPU child without the laptop. No token enters source, manifests, model artifacts or logs. The CPU child mounts only its verified W&B credential dataset. Metadata has enable_gpu=false/enable_tpu=false, bootstrap hides CUDA, the CPU runner rejects usable CUDA.

Use immutable artifact versions, confirm checkpoint algorithm before evaluation, retain training results/checkpoints on submission/render failure, and record CPU job URL. A failed learner must not block visualization of an independently successful learner. Time-budget termination still saves and visualizes the partial-budget policy with its completed episode count; do not label that as 500 completed episodes.

Verification: baseline suite, TDD tests for sampled animation/validation and actual outbound CPU metadata/handoff failures, full CPU suite, independent review, then one real short GPU-to-CPU online smoke. Only after that succeeds, publish and launch the selected long trial with checked quota. Final report distinguishes old W&B resource samples from vmap benchmark measurements and future long-run telemetry.
