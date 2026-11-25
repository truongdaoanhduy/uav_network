# Local verification of reference v3.4 — 2026-11-25

Source archive: `zhang_xue_uavnetsim_v3_2_verified_20261125.zip` SHA256 `52a1472bd5d7254f0b8ce50269db7824f86803c4da2d69b703cae931c3d92c89`.

- RED: 7 initial regressions failed against the original v3.2 mini adapter (5 assertion failures, 2 errors).
- GREEN: 14/14 reference `unittest discover` checks.
- Gate 4 CPU toy actor: motion grad 0.4889, discrete destination grad 0.0262, blackout entropy mask safe.
- Existing production subset on isolated worktree: 30/30 slotted/M2 related `pytest` checks.
- Runtime invariant sweep: 20 independent seeds × 100 packet attempts, PDR in [0,1], bounded queues, DS masses finite.
- Host Python `/home/aduy/Documents/NCKH/uav_network/.venv/bin/python`: torch 2.7.1+cu126 but `torch.cuda.is_available()==False`.
- Deferred: actual SimPy UavNetSim PHY/MAC integration, Warp parity (Gate 5), six-agent joint actor/critic training, Kaggle, W&B/HF.
- Security: Sandbox MCP server currently reports `auth_enabled=false` and Cloudflare quick tunnels running. Authentication/protection is NOT remediated in this code branch.

This is a reference package and is **not** proof of production integration.
