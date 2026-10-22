# CPU replay: unfinished task brief

Status: proposal investigated; implementation and full-buffer validation pending.
Read [CURRENT_TASK.md](CURRENT_TASK.md) for running jobs and constraints.

## Intended outcome

Test whether a 1,000,000-row RL replay buffer can live in CPU RAM while keeping
the simulator/collector on GPU0 and learning on GPU1 on a two-T4 machine.
Preserve batch 4096 and existing algorithm/scientific behavior.
The memory estimate is ~12.21 GiB for replay and ~51.16 MiB of raw sampled
transition data per batch; these are not measured peak host memory.
Verify actual Kaggle RAM/VRAM and quota rather than assuming 30 GB is available.

This change targets the MASAC PQL 1M CUDA OOM. It does not establish that 1M
learns better than 500k or solve the collector bottleneck.

## Read the existing flow first

- `uav_network/algorithms/replay.py`: TensorReplayBuffer already accepts
  `device="cpu"`. Its `pin_memory` currently pins full storage as well as
  sampled batches. Avoid locking the whole ~12 GiB buffer.
- `uav_network/experiments/training.py`: BASE already resolves
  `runtime.replay_device` and passes CPU replay to the learner.
- `uav_network/experiments/pql_system.py`: PQL currently constructs
  MeasuredReplay on `self.critic_device`, ignoring that runtime setting.
- `uav_network/experiments/pql_replay.py`: MeasuredReplay overrides sample
  and checkpoint metadata. Its PQL state currently lacks an explicit
  replay device type.
- `uav_network/algorithms/optimization.py`: tensor_batch already moves
  sampled tensors to learner device with `non_blocking=True` for CUDA.
- `uav_network/experiments/pql_workers.py`: freeze_transition clones on the
  source device. Read critic and policy worker batch ownership/lifetimes;
  do not invent an extra GPU1 replay copy.
- `uav_network/experiments/fixed_checkpoints.py`: replay_state clones CPU
  copies; restore_replay requires the saved RNG device type. At 1M,
  snapshot and eager restore copies can dominate RAM.
- `uav_network/experiments/pql_checkpoints.py` and
  `uav_network/algorithms/checkpoints.py`: validate before mutation;
  preserve old checkpoint compatibility and explicit continuation labels.

## Continuation checklist

- [ ] Recover the design stage and current branch state. Classify this existing
  replay path with Superpowers, honoring prior user authorization and any
  applicable review gate; do not claim an unreviewed artifact was approved.
- [ ] Start from public fix `ab60690` (this handoff branch already contains it);
  keep Gen1. Use an isolated feature branch for implementation.
- [ ] Run the relevant baseline tests. Read
  `tests/test_measured_replay.py`, `tests/test_pql_system.py`,
  `tests/test_optimization.py` and existing checkpoint tests located with
  `rg -n 'replay_state|restore_replay|replay_device|pin_memory' tests`.
- [ ] Write and observe RED for PQL CPU replay placement and preserved
  default/auto behavior. Then minimally honor `runtime.replay_device`.
- [ ] Write and observe RED for bounded batch pinning/transfer behavior,
  including PQL's sample override. Keep full CPU storage pageable and GPU
  allocations bounded to live minibatches; ensure async copy source lifetime.
- [ ] Verify CPU replay same-device save/load restores next sample RNG,
  cursor, measurement metadata and learning state. Reject unsupported
  CPU/CUDA RNG changes before mutation; a reseed is not exact continuation.
- [ ] Measure/save/load peak CPU RAM at a drained checkpoint boundary.
  Preserve immutable snapshot behavior and avoid full-size unnecessary
  copies. Evaluate mmap loading where compatible; test that updates to a
  live replay cannot corrupt a checkpoint.
- [ ] Run focused tests RED→GREEN and the full project test command.
  Record actual command/results and any skipped CUDA checks. Commit and push
  before submitting remote validation.
- [ ] Run **one bounded separate** real two-T4 validation job after checking
  existing jobs and quota. Keep env=2048, batch=4096, replay capacity=1M,
  chosen AMP/compile/fused settings pinned and Gen1. A fresh job is not an
  exact resume of a GPU replay checkpoint.
- [ ] Exercise a **fully occupied** replay of 1,000,000 rows, actual learner
  updates including compilation peaks, and checkpoint save/load/resumed
  updates. Capture peak host RSS/available RAM, per-GPU allocated/reserved
  peaks, transition/sample/transfer/update times, loss finiteness, counters,
  commit/config and exit status in JSON. Use MASAC PQL first because it OOMed.
  Label synthetic or duplicated rows as memory-test data if used to fill
  cheaply; draw no policy-quality conclusions from them.
- [ ] Inspect logs/result JSON. Record passed/failed/not run separately and
  update the ledger. Recommend production adoption only from these results;
  preserve existing 500k jobs while the gate is pending.

## Verification entry points

Use the project's available Python environment, checking imports first:

```bash
python3 -m pytest -q tests/test_measured_replay.py tests/test_pql_system.py tests/test_optimization.py
python3 -m pytest -q
git diff --check
git status --short
```

The server is CPU-only with about 7.7 GiB RAM. Use small local test buffers;
do not allocate a full 12 GiB replay or eagerly load an 8–13 GB checkpoint
there. Full-capacity evidence belongs on the bounded Kaggle validation job.

PyTorch primary documentation retrieved with Context7:
https://github.com/pytorch/pytorch/blob/main/docs/source/notes/cuda.md
and https://github.com/pytorch/pytorch/blob/main/torch/_tensor_docs.py .
Pinned buffers can accelerate host→GPU transfer; excessive pinned storage
can exhaust RAM. Cross-stream use and pinned-source mutation require
completion/synchronization. Reuse existing Torch facilities; add no dependency.

## What counts as completion

Implementation is complete only with fresh local verification and a recorded
code commit. The Kaggle memory gate is separately complete only when full
occupancy, real updates and checkpoint/resume evidence are present.
Do not turn allocated capacity, a 10,240-row microprobe, a submitted job,
or W&B `finished` into proof of full-buffer stability.
