# Zhang–Xue–UavNetSim v3.4 — repaired **reference** package

**Status (2026-11-25): local reference simulator and unit checks; NOT integrated into the production UAV environment, and NOT a full UavNetSim SimPy implementation.** This folder derives from the user's original `zhang_xue_uavnetsim_v3_2_verified_20261125.zip`; the unmodified original readme is preserved as `V3_2_README_ORIGINAL.md`.

## Tested modifications

- `PacketQueueManager` limits **transmitting + waiting** together, never schedules rejected packets, and releases capacity when a packet leaves the transmit queue.
- `MiniDiscreteEventNet` retries waiting packets when a route/radio recovers on the next `run_until` call. A failed RX caused by blackout is parked at its previous hop. Every retry retains the original application deadline; data PDR excludes E2E ACKs. This is a simplified deterministic protocol test harness, not CSMA/CA.
- `SearchDataPacket` requires immutable tuple payloads, validates on-wire ranges with real `struct.pack`, and rejects invalid time/TTL. `send_data` rejects own destination and nonexistent peers.
- `RewardEngine.compute_unified_step_reward` now interprets safety/boundary inputs as **nonnegative costs and subtracts them**. Its **R1-only** information credit accepts per-cell vectors, projects onto true target cells in the **environment-only reward**, and does not credit false-positive information. Nonzero scalar information gains now raise `ValueError`, because attribution is impossible; zero scalars remain allowed. This is an intentional interface change. **Do not expose the target truth mask to actors.**
- Original v3.2 configuration, R0 interpretation, DS masses, hybrid action example and its tests remain available.

## Verified commands

Use a Python interpreter with NumPy, PyYAML, and PyTorch:

```bash
cd /home/aduy/Documents/NCKH/uav_network_zhang_xue_v34_research_20261125/research/zhang_xue_v34_reference
/home/aduy/Documents/NCKH/uav_network/.venv/bin/python -m unittest discover -s tests -v
/home/aduy/Documents/NCKH/uav_network/.venv/bin/python uav_adapter/gate4_hybrid_marl_test_template.py
```

Fresh observations: **14/14 reference unit tests passed**, tiny **Gate 4 PyTorch CPU gradient probe passed** (motion_grad ≈ 0.4889, destination_grad ≈ 0.0262), **30/30 existing production slotted/M2 selected tests passed**, and a **20-seed × 100-packet invariant sweep passed**. This is not a GPU convergence or full simulator benchmark.

## Explicitly not completed

- **No** replacement of production `uav_network` environment, learner/replay/action pipeline, or UavNetSim network adapters.
- **No** parity with actual SimPy or GPU Warp PHY/MAC: Gate 5 remains deferred as requested.
- **No** verified application E2E retransmission timer, dirty-delta anti-entropy, fully dynamic event-time topology, or inter-agent actor observation audit.
- **No** real CUDA test on this Ubuntu session (`torch.cuda.is_available() == False`), Kaggle training, W&B/HF mutation, or GitHub push.
- `config_v3.2.yaml` is a preserved *reference proposal*, **not** a user-approved final frozen scenario.
- MiniDiscreteEventNet's queue and RX-failure logic is deliberately simplified. UavNetSim's actual `feed_packet`, MAC ACK, SimPy buffer, routes, and delays still require integration tests.

## Source references

- UavNetSim-v1 upstream: https://github.com/Zihao-Felix-Zhou/UavNetSim-v1
- PyTorch ST Gumbel-Softmax implementation: https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py
- Zhang T-IV 2024: https://doi.org/10.1109/TIV.2024.3352581
- Zhang Drones 2024: https://doi.org/10.3390/drones8080393
- Xue 2026: https://doi.org/10.3390/electronics15091875

Next: integrate receiver-local DS and its own-only packet transport into the real production step pipeline *without oracle leakage*, make a true hybrid actor/replay update on production dimensions, and validate per-event packet metrics; only then revisit deferred parity and full training.
