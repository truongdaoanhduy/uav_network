# GPU-to-CPU Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train500 episodes/algorithm with1000-step deadlines, then visualize immutable checkpoints in a new CPU-only Kaggle job.

**Architecture:** Extend the existing evaluator and Kaggle launcher. Keep model/SimPy code unchanged; W&B artifacts transfer policy checkpoints, and the GPU runner submits its private CPU child before returning.

**Tech Stack:** Python, PyTorch, Matplotlib/Pillow/ffmpeg, existing Kaggle CLI and W&B SDK.

**Spec:** `docs/superpowers/specs/2026-10-14-gpu-cpu-visualize.md`

## Global Constraints

- Native UavNetSim revision and all actor/critic independence remain unchanged.
- Long profile: max_steps1000, episodes500 total per algorithm, num_envs2, batch512, replay100000, warmup4000, 1update/vector step, seed44/eval10044.
- Every Python file remains at most500 lines.
- CPU metadata disables both GPU/TPU; credentials are never included in source/artifacts/logs.
- Existing default evaluate/launch behavior remains supported.

## Review Focus

- Terminal animation frame survives non-divisible frame strides.
- Missing/unuploaded or aliased checkpoint does not trigger a misleading visualization.
- Public/unowned credential datasets are rejected before use.
- Successful sibling results survive failed training/upload/submission.
- Soft time budget cannot be reported as completed episode budget.

### Task1: CPU animation and long profile

**Files:** Modify experiments/evaluation.py, experiments/rendering.py, experiments/cli.py; create configs/motion-long.yaml and tests/test_cpu_visualization.py.

**Interfaces:** evaluate(checkpoint, episodes=None, output_dir=None, seed=None, animate=False, frame_stride=5, fps=10, video_format="auto") returns existing metrics plus animation path/metadata.

- [ ] Write tests that train/save a real small native policy, evaluate it on CPU and decode its GIF; assert steps0,3,6,7 and4 frames. Reject stride/fps0 and invalid format before output.
- [ ] Observe failure, add minimal snapshot/animation code and CLI options.
- [ ] Validate long profile scientific differences and run targeted tests.

### Task2: Private CPU-only cloud handoff

**Files:** Modify cloud/client.py, cloud/kernel.py, cloud/cli.py, cloud/remote.py; create cloud/visualize.py and tests/test_cpu_handoff.py.

**Interfaces:** build_visualization_spec(parent_spec, results) yields immutable artifact mapping; submit_visualization(parent_spec, results, output) writes a public CPU launch manifest and returns it; run_visualization(spec, output) evaluates downloaded checkpoints on CPU.

- [ ] Write boundary tests for CPU metadata, immutable artifact/algorithm validation and preservation of saved results on handoff failure. Mock only external Kaggle/W&B transport.
- [ ] Observe failures, extend existing private-dataset verification; prepare only the selected account's private launch-token dataset.
- [ ] Make GPU finalize skip inline evaluation in the new mode, submit CPU child after model upload, and preserve legacy mode.
- [ ] Execute targeted tests then full CPU suite; request independent review.

### Task3: Real cloud verification, publication and long run

**Files:** README.md and docs/gpu-cpu-flow-20261014.md/evidence.json.

- [ ] Commit/publish tested code, run a tiny online GPU train that automatically launches CPU-only rendering, and inspect actual artifacts and metadata.
- [ ] Correct observed integration failures and repeat affected checks if needed.
- [ ] Publish verified evidence, launch500/1000 trial with adequate quota, preserve returned W&B/Kaggle URLs.
- [ ] Report settings, measured resource samples and the meanings of vmap/packet-network porting.
