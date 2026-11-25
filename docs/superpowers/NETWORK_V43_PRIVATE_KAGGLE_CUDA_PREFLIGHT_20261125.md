# Zhang–Xue v4.3 — Account2 CUDA test packaging and launch checkpoint

Date 2026-11-25. Branch experiment/zhang-xue-v43-kaggle-cuda-preflight-20261125, parent v4.2 ee58650f389be72b547aa381ea68e417f4591a15.

This is a CUDA **preflight**, not training. Local Ubuntu lacks an accessible NVIDIA CUDA driver. A separate ephemeral venv for Kaggle CLI 2.2.5 exists at /tmp/zhang-xue-kaggle-cli-new-v42, independent of installed v1.7.4.5. Account2 in the external private key.txt uses new KGAT-style token; new CLI authenticates using KAGGLE_API_TOKEN environment variable; quota check reported 29.53h remaining at inspection. No secrets/credentials are embedded in source, metadata, artifacts, notebook or repository.

Generator scripts/build_zhang_xue_v43_kaggle_cuda_smoke.py packages ONLY a strict list of 12 research Python source files from immutable v4.2 commit via git show, with SHA256 manifest, ZIP/base64 payload, and private Kaggle metadata. Kaggle runner checks NVIDIA Warp 1.18.0 on cuda:0 and cuda:1, compares E16 CPU/GPU, and tries E1024 and E4096 with full source-local Warp map→queue→2-hop DATA→ACK preflight. All operation results are written to Kaggle output zhang_xue_v43_cuda_probe.json. It cannot update W&B/HF or launch MATD3/MASAC.

Local test tests/test_zhang_xue_v43_kaggle_cuda_bundle.py verifies package contents, private metadata, source SHA and no credentials. The GPU job must be **executed** on Kaggle to claim any GPU success. Never interpret submission success as execution success.

Remaining research Gate5 issues: full 6-UAV contention and source-exact SimPy CSMA, persistent route control/ACK in all 200 scenario steps, GPU-host synchronization profiling and packet buffering under real offered load, complete actor→reward and no-oracle belief evaluation. **DO NOT train until fully checked.**
