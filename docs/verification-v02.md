# Verification: battery, MATD3 and parallel environments

Date: 2026-10-12. Repository: `/home/aduy/Documents/NCKH/uav_network`.
Host: Ubuntu/Linux, Python3.13.2, PyTorch2.7.1+cu126, Gymnasium1.3.0,
8 logical/available CPUs. `torch.cuda.is_available()` is **false**.
UavNetSim HEAD: `04daafb815eb377409b40b285574eeb62b9a8d58`.

## Automated and real-network checks

```bash
.venv/bin/python -m unittest discover -s tests -v
```

56 CPU checks pass; 3 dedicated real-CUDA checks are explicitly skipped on this
host. They test immediate CPU consumption after GPU replay D2H copies, pinned
H2D transfer and both CUDA learners training from spawned real network workers.
Run them on the workstation with `-p test_cuda.py`; skipped tests are not GPU evidence.

Covered behavior:

- Funded motion cost, observed charge, safe stopping against newly exhausted
  peers, dead sensing, all-empty failure versus timeout truncation.
- Native RF debit once per attempt, retries/failed frames, GCS energy separation,
  unaffordable frames blocked and dead receiver custody rejected.
- Real direct GCS delivery, policy-selected custody relay, fragmentation,
  original TTL, contention/duplicate-head regressions and same-seed network.
- Peer obstacle LOS→NLOS lower gain; extra10 dB gives0.1× NLOS gain and no LOS change.
- Six-UAV integration with one obstacle: blocked 0→1 fails and retains report;
  hand-selected 0→2→3→GCS reaches GCS in3 mission hops while every UAV spends pin.
  This verifies action/network behavior, not learned routing convergence.
- Both actor branches/critics receive gradients; MATD3 actors and targets wait
  for policy_delay; masked batched actions, silent power0 and inactive movement0.
- Checkpoint exact deterministic actions, algorithm mismatch/version/dimension
  validation; policy parameters must be finite before save.
- Separate real network worker PIDs, shared local/state data, terminal state in
  replay before masked reset, unchanged continuing-slot state and worker cleanup.
- Final warmup-reaching collection is learned with/without overlap; every saved
  model tensor matches between those one-batch runs.

## Final smoke runs after review fixes

`outputs/v02-final/` holds local logs/checkpoints/evaluation; all are Git-ignored.

| Algorithm | Envs | Episodes | Joint transitions | Learner updates | Train elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: |
| MASAC | 1 | 2 | 80 | 73 | 4.265 |
| MATD3 | 1 | 2 | 80 | 73 | 3.157 |
| MASAC | 2 | 4 | 160 | 77 | 3.925 |
| MATD3 | 2 | 4 | 160 | 77 | 3.069 |

All six-UAV runs keep real networking and2 obstacles enabled. Parallel runs
exercise partial resets. Each checkpoint evaluated successfully on CPU and
produced metrics/PNG with charge labels. Held-out tiny-smoke search/delivery
rates were0%; these checks prove execution/gradients, not convergence or quality.
Different update/sample budgets mean this table is not an algorithm speed ranking.

## Collection benchmark

```bash
.venv/bin/python -m uav_network benchmark --config configs/smoke.yaml \
  --env-counts 1 2 4 --steps 64 --learner-updates 10 \
  --output outputs/v02-final/benchmark.json
```

| CPU env processes | Joint transitions measured | Joint transitions/s | Warmed learner updates/s |
| --- | ---: | ---: | ---: |
| 1 (sync) | 64 | 88.71 | 36.09 |
| 2 (spawn) | 128 | 159.92 | 35.89 |
| 4 (spawn) | 256 | 240.06 | 38.94 |

One timing sample per configuration, batch8/hidden[32,32], real6-UAV/2-obstacle
network. Collection includes random valid actions and worker waits; excludes
replay writes, startup and resets. Learner rates use2 warmup +10 measured updates.
This measures collection scaling on the connected CPU, not RTX6000 training.
Workstation batch1024/hidden256 workload and CUDA/compile performance remain unmeasured.

## Independent review and setup

Important findings reproduced and fixed:

1. Overlap discarded the final deferred update: drain the final ready group.
2. Zero configured entropy targets accidentally froze temperatures: use actual
   active/control-choice availability instead of target values as a sentinel.
3. Asynchronous CUDA→CPU copies could expose unfinished CPU data: block D2H.
   This is supported by [PyTorch's transfer guidance][transfer]; no GPU reproduction
   was possible here. Dedicated CUDA regression covers it on a CUDA host.

Minor review corrections: advancing benchmark reset seeds and inactive sensory
slots. Final independent static review found no remaining Critical/Important issue.
Reviewer additional checks passed1000 randomized funded-motion/safety cases,
an exactly affordable native frame, retry budget exhaustion and receiver death
during a frame. Final executable checks remain CPU-only on this host.

`bash setup.sh --system-site-packages` reran successfully, installed project0.2.0,
confirmed the pinned upstream and Gymnasium DISABLED autoreset capability.
Owned source-size contract is checked by `test_project_structure.py`.
Generated results/dependencies stay excluded from Git.

[transfer]: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
