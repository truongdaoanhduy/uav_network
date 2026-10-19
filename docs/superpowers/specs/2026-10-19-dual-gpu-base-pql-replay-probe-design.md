# Dual-GPU BASE/PQL Replay Probe Design

## Goal
Use both Kaggle T4 GPUs for both BASE and PQL replay-capacity probes, then find the largest replay capacity that passes `allocate + collect + sample + critic update + actor update` at `num_envs=4096` and `batch_size=8192` over the sequence `5M -> 4M -> 3M -> 2M -> 1M`.

## Constraints
- Do not change MASAC/MATD3 mathematics, reward, scenario, network model, or action semantics.
- Do not restart or mutate the four canonical full runs.
- Keep PQL's existing role split: GPU0 environment/collector, GPU1 replay + critic/policy learner.
- Add the same role split to BASE: GPU0 Warp environment + inference collector, GPU1 replay + learner.
- A claimed dual-GPU pass must show both GPUs are visible and materially used; `CUDA_VISIBLE_DEVICES=0,1` alone is not evidence.
- Replay remains unsharded in this change. A replay OOM on GPU1 is a real failure; do not silently split capacity or lower batch size.
- Test order is 5M, then 4M, 3M, 2M, 1M; stop only after collecting enough evidence to identify the largest passing capacity for each mode.

## BASE Architecture
BASE gets explicit collector and learner devices derived from visible CUDA devices. With one visible GPU, both roles stay on `cuda:0` for backward compatibility. With two or more visible GPUs, Warp environment and inference actor live on `cuda:0`, while the authoritative learner and replay live on `cuda:1`.

The collector is a read-only inference copy of the learner actors. After learner actor updates, the collector receives the newest actor weights at a collection boundary. Transitions cross GPU0 -> GPU1 when inserted into the replay; sampled batches remain on GPU1 for learner updates. No DDP/DataParallel is introduced.

## PQL Architecture
PQL keeps its current device resolver and worker topology. The probe verifies that it resolves to collector `cuda:0` and critic/policy/replay `cuda:1` on two visible T4s.

## Benchmark Evidence
Each result records mode, algorithm, replay capacity, env count, batch size, devices, replay bytes, environment bytes, critic/learner updates, actor updates, peak allocated/reserved bytes per GPU, and status (`passed`, `oom`, `failed`). A pass requires at least one critic/learner update and at least one actor update.

## Kaggle Execution
Use a pinned commit and private Kaggle jobs. BASE and PQL probes must each receive both GPUs (`CUDA_VISIBLE_DEVICES=0,1`). W&B may log benchmark summaries, but no checkpoint artifact is needed. Probe MATD3 first because replay layout is shared and it exercises delayed actor updates; if requested later, MASAC can be run at the winning capacity as a parity smoke.

## Non-goals
- Replay sharding across GPUs.
- DDP/DataParallel.
- AMP or `torch.compile`.
- Changing update intensity, warmup policy, batch size, or environment count during this capacity probe.
