# GPU packet environments on Kaggle

Use `configs/motion-gpu.yaml` for a fresh learner: 128 complete GPU environments, 1024 total episodes per algorithm, up to 1000 steps per episode, seed44. Each world still contains six UAVs. All geometry, camera sensing, report custody and CSMA/CA DATA/ACK processing run through NVIDIA Warp kernels. Torch observations/actions/replay stay on CUDA. There are zero CPU simulator workers.

The native UavNetSim backend remains the default for older configs/checkpoints. This is an explicit `warp_packet` port; it is not a renamed native simulator or the Salesforce WarpDrive package.

## Kaggle notebook

Enable a T4 GPU and internet access, then run:

```python
import os, subprocess, sys
os.environ.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1",
                  MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                  CUBLAS_WORKSPACE_CONFIG=":4096:8", MPLBACKEND="Agg")
subprocess.run(["git", "clone", "--branch", "gpu-native-20261014",
                "https://github.com/truongdaoanhduy/uav_network.git",
                "/kaggle/working/uav_network"], check=True)
root = "/kaggle/working/uav_network"
subprocess.run(["bash", "setup.sh", "--system-site-packages", "--cloud"],
               cwd=root, env=dict(os.environ, PYTHON=sys.executable), check=True)
python = root + "/.venv/bin/python"
subprocess.run([python, "-m", "uav_network", "train",
               "--config", "configs/motion-gpu.yaml", "--algorithm", "masac",
               "--set", "tracking.enabled=false"], cwd=root, check=True)
```

This writes the checkpoint to `outputs/motion-gpu/final.pt`. To run MATD3, use `--algorithm matd3` and set a fresh `runtime.output_dir`. For online W&B, use the existing private credential-dataset launcher; leave tracking enabled.

The first Warp run compiles kernels. Compilation/startup is separate from warmed training throughput.

## GPU training followed by a separate CPU session

The existing launcher already supports this handoff:

```bash
.venv/bin/python -m uav_network kaggle launch --stage train \
  --config configs/motion-gpu.yaml --algorithm both \
  --credential-dataset OWNER/YOUR_PRIVATE_CREDENTIAL_DATASET \
  --cpu-visualize
```

It uploads final checkpoints to W&B, then submits a separate CPU-only visualization job. Warp is included in the cloud installation. That CPU session loads the checkpoint's `warp_packet` backend and runs the same kernels on CPU with one world.

For a manual handoff, download `final.pt`, create a new Kaggle session with accelerator **None**, install the project, then run:

```bash
.venv/bin/python -m uav_network evaluate --checkpoint /path/to/final.pt \
  --episodes 1 --seed 10044 --output-dir outputs/visualize --animate
```

It creates PNG plus MP4/GIF according to the available encoder. CPU policy inference can produce a different trajectory from CUDA inference; this is a new seeded evaluation episode.

## Repeat the capacity sweep

```bash
.venv/bin/python -m uav_network.experiments.env_scale \
  --config configs/motion-long.yaml --backend warp_packet --device cuda \
  --counts 128 256 512 1024 2048 4096 --algorithms masac matd3 \
  --steps 16 --min-seconds 10 --case-timeout 600 \
  --output outputs/gpu-capacity-new
```

Each case uses a fresh process and the same model/batch/replay/world settings as the earlier native benchmark. The output directory must be empty. The sweep stops at its first failure; actual Torch CUDA OOM, Warp CUDA allocation failure, host RAM pressure and packet buffer overflow have distinct statuses. It retains completed measurements and identifies skipped counts.

NVIDIA-SMI VRAM includes Warp buffers as well as Torch model/replay memory. Torch allocator counters alone omit Warp allocations. GPU utilization and occupied VRAM measure different things.

## Learning budget and limits

Increasing environments supplies more transitions per vector step. With one update of batch512, sample uses per new joint transition are `512 / num_envs`: 4 at128, 1 at512, and0.125 at4096. More throughput does not establish improved convergence. Start with128 for a learning trial and compare reward/delivery curves before raising the count. Use `runtime.episodes` for the total across all worlds, not episodes per world. The learner runs the final vector batch as a whole; simultaneous completions can exceed a non-aligned requested episode count.

Environment PCG64 world generation and Python MT19937 MAC backoff streams preserve native seeded behavior in the tested corpus. GPU warmup policy actions use a dedicated Torch generator with the same distribution; entire native-versus-Warp training trajectories are not promised identical.

This port supports the project's analytical PHY with urban NLOS and max_retries<=26. Unsupported settings or exhausted internal event/packet/frame capacities fail explicitly. Geometry/energy/event clocks use FP64; observation/state/reward buffers use FP32. Native-vs-device parity is tested on scripted cases; it is not a proof for every floating-point boundary/configuration. See [primary-source research](gpu-native-env-research-20261014.md) and the measured capacity report.
