#!/usr/bin/env python3
"""Controlled PEER store-carry-forward experiment, not mission feasibility.

GCS is an application destination and may use a multi-hop route. Two separated
contacts isolate PEER dissemination from routed GCS traffic. Repositioning the
relay between contacts is an external network fixture, not simulated flight.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]


def run_peer_relay_partition_probe(scenario_path: Path, device: str, seed: int = 44) -> dict:
    """Same sensing, contacts, actions and data-mule flag; only PEER is toggled."""
    import warp as wp
    from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80
    results = {}
    cid = 34  # center (725,75), only sensed by source UAV0
    for enable_peer in (False, True):
        params = GPUWorldV80Params.from_yaml(
            scenario_path, envs=1, device=device, seed=seed,
            scenario_overrides=(
                "mac_noise_power_w=4.0e-11",
                f"enable_peer_dissemination={'true' if enable_peer else 'false'}",
                "enable_data_mule_forwarding=true",
            ),
        )
        w = WarpParallelMissionV80(params)
        w.network.reset()  # discard spawn-era routes, packets and evidence
        pos = np.array([[[725.,75.,50.], [550.,75.,50.],
                         [900.,900.,50.], [900.,800.,50.],
                         [800.,900.,50.], [800.,800.,50.],
                         [50.,50.,15.]]], dtype=np.float32)

        def place():
            wp.copy(w.buffers["positions"], wp.array(pos, dtype=wp.float32, device=device))
            wp.copy(w.buffers["pos_xy"], wp.array(pos[:, :6, :2], dtype=wp.float32, device=device))
            w._update_phy()

        def links():
            gain = w.network.data["gain"].numpy()[0]
            threshold = params.noise_power_w * w.snr_lin
            return {name: bool(params.tx_power_w * gain[i,j] >= threshold)
                    for name,i,j in [("source_to_relay",0,1),("source_to_gcs",0,6),("relay_to_gcs",1,6)]}

        place()
        truth = np.zeros((1,400), dtype=np.uint8)
        truth[0,cid] = 1
        wp.copy(w.buffers["truth"], wp.array(truth, dtype=wp.uint8, device=device))
        velocity = wp.zeros((1,6,2), dtype=wp.float32, device=device)
        # Identical offered actions: disabled PEER resolves to SILENT in control.
        dest = wp.array(np.array([[2,1,0,0,0,0]], dtype=np.int32), device=device)
        topology_a = links()
        for _ in range(25):
            w.step(velocity, dest)
        relay_count = int(w.network.data["ledger_pos"].numpy()[0,1,0,cid])
        phase_a_tp = int(w.buffers["gcs_tp"].numpy()[0,cid])
        pos[0,1,:2] = [225.,75.]
        place()
        topology_b = links()
        for _ in range(25):
            w.step(velocity, dest)
        wp.synchronize_device(device)
        key = "peer_enabled" if enable_peer else "direct_gcs_only"
        results[key] = {
            "label": "peer_enabled" if enable_peer else "peer_disabled_control",
            "gcs_tp_confirmed": int(w.buffers["gcs_tp"].numpy()[0,cid]),
            "phase_a_gcs_tp": phase_a_tp,
            "relay_source_positive_count_before_contact_b": relay_count,
            "source_count_received_at_gcs": int(w.network.data["ledger_pos"].numpy()[0,6,0,cid]),
            "packets_to_gcs": int(w.network.data["stats_gcs"].numpy().sum()),
            "packets_to_peer": int(w.network.data["stats_peer"].numpy().sum()),
            "overflow": int(w.network.data["overflow"].numpy().sum()),
            "failed": int(w.buffers["failed"].numpy().sum()),
            "contact_a": topology_a, "contact_b": topology_b,
        }
    a,b = results["direct_gcs_only"],results["peer_enabled"]
    results["peer_relay_unlocks_gcs_confirmation"] = bool(
        b["gcs_tp_confirmed"] == 1 and a["gcs_tp_confirmed"] == 0
        and b["relay_source_positive_count_before_contact_b"] > 0
        and b["source_count_received_at_gcs"] > 0
        and a["overflow"] == b["overflow"] == a["failed"] == b["failed"] == 0
        and b["contact_a"] == {"source_to_relay":True,"source_to_gcs":False,"relay_to_gcs":False}
        and b["contact_b"] == {"source_to_relay":False,"source_to_gcs":False,"relay_to_gcs":True}
    )
    results["scope"] = "Controlled 1-target network fixture; relay reposition is not flight feasibility evidence"
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--scenario", type=Path, default=ROOT / "research/zhang_xue/configs/scenario_v91_bayes_1km.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--output", "--out", type=Path, required=True)
    args = parser.parse_args()
    proof = run_peer_relay_partition_probe(args.scenario, args.device, args.seed)
    passed = proof["peer_relay_unlocks_gcs_confirmation"]
    out = {
        "status": "PASS" if passed else "FAIL",
        "scenario_sha256": hashlib.sha256(args.scenario.read_bytes()).hexdigest(),
        "seed": args.seed, "device": args.device,
        "dcf_backoff_scope": {
            "warp_range": "0..CW (CWmin=31: 32 choices)",
            "upstream_range": "0..CW-1 (CWmin=31: 31 choices)",
            "verdict": "Range convention only; not full IEEE conformance or UavNetSim equivalence",
        },
        "peer_relay_partition_proof": proof, "scenario_frozen": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, allow_nan=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
