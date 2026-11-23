# M2 Target-only 2D Belief Map — logic audit and regression fixes (21 Nov 2026)

## Frozen mission semantics

- World `1000 m × 1000 m`; 40×40 2D target-belief grid (`F`,`E`,`U`, cell side 25 m); 6 UAVs, 35 targets, 25 physical obstacles. UAVs move in 3D for sensing, collision avoidance and radio; no reconstructed 3D geometry or 2D obstacle map. Simulated obstacle geometry is used by the **CBF safety shield** and camera occlusion only; this is an oracle simulator shield, not a learned local obstacle map.
- No ground-truth target occupancy enters actor/critic observation, GCS packet payload, or DS map; truth is used by the sensor simulator and evaluation/reward only.
- Packet-delivered event IDs are de-duplicated; incoming evidence can be out-of-order. GCS only merges packets physically delivered through Native UavNetSim; its F/E/U map can be sent back via `MAP_DELTA` to a reachable UAV, then relayed by policy-selected hops.
- Stage remains Native CPU reference. Warp 4096 environment training, full MATD3/MASAC policy convergence and Kaggle UI sessions remain deferred. API token is not used or modified for this audit.

## Reproduced failures and concrete fixes

1. **Decision revision reused after retraction/re-discovery:** original `CooperativeWorldMap` forgot the last revision when the candidate vanished and restarted at revision 1. Remote GCS can drop a later genuine claim as stale. Persist last revision per cell even when candidate absent. Test `test_claim_revisions_never_reset_after_candidate_retracted_and_rediscovered` failed before patch.
2. **Evicted relay custody never recovered:** receiver tracks unique message ID even after its forwarding copy has been evicted. On a subsequent physically delivered copy from another peer, it discarded the payload and could no longer relay. Restore the buffer copy after valid receipt but do not re-fuse evidence. Test `test_reacquired_packet_restores_forwarding_custody_after_eviction_without_recounting` failed before patch.
3. **Report reward paid while GCS disputes the claim:** previous reward iterated received claim history and paid +50 for an unrevoked *message*, regardless of the currently disputed consensus. Now +50 is paid once per **currently verified non-disputed correct target cell**, and the same current-consensus condition gates one-time false-report penalty; waiting for a later correction pays correctly on resolution. Test `test_gcs_report_reward_waits_for_consensus_and_is_not_paid_twice` failed before patch.
4. **Tiny FoV sliver crashes sensor evidence:** direct `1-(1-Pf)^q` rounds to zero for nonzero `q≈1e-16`, violating `0<p0<p1<1`. Use `-expm1(q log1p(-Pf))`. Test `test_partial_fov_tangency_likelihoods_are_stable_even_for_tiny_nonzero_overlap` failed before patch.
5. **O(n²) repeated correlated DS likelihood replay:** 200 chronological cell observations previously invoked `SharedBiasLikelihood.observe` 20,100 times (triangular sum). Added cached chronological stream likelihoods and support totals, with **sorted exact replay only for late/out-of-order events**. New test requires ≤240 observations; original failed. Randomized delayed packets with seeds 1, 7, 42, 100 verify identical final normalized F/E/U and verification status.
6. **Caller Config silently mutates from 4D to M2 5D:** constructing an opt-in M2 environment previously set `config.runtime.m2_enabled=True` on the caller's object, affecting later legacy training. Deep-copy before enabling M2 and make learner tests use `env.config`. Test `test_m2_environment_does_not_mutate_caller_config_or_legacy_action_abi` failed before patch.
7. **Actor did not see its candidate/verified/rejected status:** `visible_summary` calculated these statuses but omitted them from the observation patch. Added 3 status bits per local cell so the Actor can distinguish verification stages. This **intentionally changes M2 observation dimensions**; no old M2 learner checkpoint should be loaded without migration. Test `test_actor_local_patch_distinguishes_candidate_pending_verified_and_rejected` failed before patch.
8. **Packet-size understated serialized envelope:** example evidence packet declared 157 bytes while equivalent complete JSON body was 241 bytes, excluding a safety margin. Map deltas and notices had similar fixed-header assumptions. Declared `size_bytes` now accounts for serialized envelope fields and a small padding margin; event batching reserves realistic envelope metadata; 1024-byte limit covered by tests `test_declared_radio_packet_bytes_cover_serialized_evidence_notice_and_gcs_map`, `test_large_evidence_event_ids_are_packetized_by_actual_wire_budget`.
9. **Repeated GCS delta to already ACKed UAV:** a subsequent new `MAP_DELTA` message could send the same source events to the same recipient after earlier physical ACK. GCS now tracks acknowledged evidence IDs *per destination* and avoids repeat downlink to that UAV while still admitting deltas to other UAVs. The recorded ACK only arises after transfer, never by inspecting private UAV maps. Test `test_gcs_does_not_resend_already_acknowledged_target_evidence_in_new_delta` failed before patch.

## Evidence of correct interfaces and scope

- Local targeted M2/algorithm/test suite: **156 passed, 23 subtests passed** (final test source after all changes).
- Two real training smoke traces with 6 UAVs, 1000m, 35 targets, 25 obstacles, shared+LN 256×256 networks, 5 continuous actions: both MATD3 and MASAC collected transitions and performed two successful CPU neural updates. This is **NOT convergence**.
- Native 100-step diagnostic seed 44: GCS coverage **0.13375**; **1,316 GCS events**, **2 verified correct target reports**, zero false verified reports, mean DS unknown mass ≈0.88457, total mission energy ≈98,612 J. This scripted policy result is neither trained nor statistically comparable to RL baselines.
- Scientific baseline is DOI `10.1109/TIV.2024.3352581`; unlike the paper's fixed mass 0.8 and globally connected UAV map, this experiment uses altitude/Pd/Pf-calibrated correlated conditional evidence, packet-level GCS report, and two-way synchronization.

## Pending scientific and engineering limits (do not overclaim)

- Native/UavNetSim transports **immutable Python event envelopes whose byte size and full packet delivery are simulated**; this is not a separately deployed socket/protobuf wire protocol. Retransmission, TTL and failure still depend on actual Native UavNetSim packet events.
- Sensor shared-bias model, candidate threshold and SPRT α/β are research choices, not guarantees calibrated against a real camera. Verification quality/false-report rates need multi-seed full episodes.
- Continuous actor type action is threshold-mapped to discrete packet types; continuous Q gradient does **not** establish optimal categorical packet choice. Later test proper hybrid categorical policy heads / ablation.
- Bounded per-UAV pending packet buffers do **not** imply bounded long-term map/event history. The Native reference keeps all original events for provenance and exact reordered evidence replay. Warp 4096 requires sparse bounded state, verified summary/checkpoint representation and parity checks.
- GCS can only downlink to in-range peers. Map convergence under network partitions and TTL is **not assured**. Known ACK only prevents repeated GCS downlink; it does not provide global knowledge of peer-private buffers.
- Current scene collision shield uses privileged simulator geometry. Obstacle mapping was intentionally removed at user's request, so don't claim learned obstacle mapping or SLAM.
- Repository-wide legacy checkpoint/source-contract tests may still fail. Compare with the clean parent commit `2c227ee` in an identical environment, not the dirty original worktree or a checkout without `third_party/UavNetSim`.

## Reproduce locally

```bash
cd /home/aduy/Documents/NCKH/uav_network_m2_belief2d_20261121
# Requires pinned third_party/UavNetSim mounted at this path
python -m pytest -q tests/test_m2_logic_audit_20261121.py tests/test_m2_target_belief2d_20261121.py tests/test_m2_hardening_20261121.py tests/test_m2_mission_native.py tests/test_m2_uavnetsim_integration.py tests/test_correlated_m2_protocol.py
python scripts/research/run_cooperative_m2_native_demo_20261121.py --steps 100 --seed 44
```

`key.txt` intentionally excluded from source snapshots. Account 1 was not invoked: CPU tests cover the Native reference; Warp/GPU work remains deferred at user request.

## Final whole-repository regression comparison (same pinned dependency)

The clean parent commit `2c227ee` (with `third_party/UavNetSim` installed) yielded **1514 passed, 10 failed, 105 skipped, 1713 subtests passed**. The audited M2 source yielded **1535 passed, 10 failed, 105 skipped, 1717 subtests passed**. All ten failed test identifiers are **identical** across clean parent and audited source, i.e. **zero new repository-test regression identifiers**. The source code was unchanged after this full-suite run; one supplementary target-obstacle physical occlusion unit test was added and passed, raising the final focused suite to **156 passed + 23 subtests**. These ten historical failures are **NOT** claimed fixed; they concern legacy checkpoint and source-contract checks.
