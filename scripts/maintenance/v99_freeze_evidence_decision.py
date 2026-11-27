#!/usr/bin/env python3
"""Fail-closed, reproducible scenario-freeze evaluator; never creates a Git tag.

Use V98 JSONs as historical evidence, but do NOT treat a script's own PASS
as evidence for upstream event-level parity, cumulative endurance, or ablation
causality. A scientific freeze requires independently verifiable fields.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def evaluate(mac: dict, endurance: dict, ablation: dict) -> dict:
    reasons: dict[str, list[str]] = {x: [] for x in ('S12', 'ENDURANCE', 'ABLATION')}
    s12 = reasons['S12']
    if mac.get('status') != 'PASS':
        s12.append('S12 original scientific gate not PASS')
    packet = mac.get('upstream_packet_trace', {})
    if not isinstance(packet, dict):
        packet = {}
    if packet.get('verified') is not True:
        s12.append('no actual upstream UavNetSim SimPy packet-event trace comparison')
    if not packet.get('upstream_commit_sha'):
        s12.append('missing pinned upstream UavNetSim commit')
    if int(packet.get('compared_events') or 0) < 1000:
        s12.append('insufficient matched packet events')
    event_metrics = packet.get('metrics_equivalent') or {}
    for k in ('mac_ack', 'retry', 'drop', 'delay', 'pdr', 'hidden_terminal'):
        if not isinstance(event_metrics, dict) or event_metrics.get(k) is not True:
            s12.append(f'upstream packet-level equivalent metric not verified: {k}')

    ene = reasons['ENDURANCE']
    if endurance.get('status') not in ('PASS', 'PASS_STICKY_RESET_MODE'):
        ene.append('endurance executable did not report PASS / PASS_STICKY_RESET_MODE')
    if int(endurance.get('envs') or 0) < 4096 or int(endurance.get('steps') or 0) < 900:
        ene.append('endurance did not cover 4096 env * 900 transitions')
    if endurance.get('device') != 'cuda:0':
        ene.append('not measured on cuda:0')
    for k in ('ever_safety_failure_worlds', 'ever_overflow_worlds', 'nonfinite_positions_all_steps', 'nonfinite_reward_all_steps'):
        if k not in endurance:
            ene.append(f'no cumulative all-step metric: {k}')
        elif endurance[k] != 0:
            ene.append(f'cumulative fault {k}={endurance[k]}')
    if endurance.get('observed_steps', 0) < int(endurance.get('steps') or 0):
        ene.append('all steps not monitored')
    if (endurance.get('peak_vram_mib_actual') or 0) <= 0:
        ene.append('no actual Warp/CUDA VRAM measurement (Torch allocation metric insufficient)')
    reward_err = endurance.get('max_reward_sum_err')
    if reward_err is None or float(reward_err) >= 1.0e-4:
        ene.append('reward ledger failed or was not measured')
    if endurance.get('terminal_worlds_handled') is not True:
        ene.append('terminal/truncated worlds not stopped or auto-reset in vectorized endurance')

    abl = reasons['ABLATION']
    if ablation.get('status') != 'PASS':
        abl.append('ablation executable did not report PASS')
    seeds = ablation.get('seeds') or []
    if len(set(seeds)) < 5:
        abl.append('fewer than 5 independent seed groups for confirmatory ablation')
    if ablation.get('paired_world_analysis') is not True:
        abl.append('no paired per-world effect-size / uncertainty analysis')
    if ablation.get('peer_action_necessity_confirmed') is not True:
        abl.append('PEER action necessity not established under matched network counterfactual')
    if ablation.get('coverage_term_matched_only_acknowledged') is not True:
        abl.append('DS-vs-Bayes only has matched exploration component, not identical full rewards')
    if ablation.get('battery_energy_accounting_verified') is not True:
        abl.append('battery energy integration and termination not independently verified')
    cases = ablation.get('results') or {}
    required_cases = ('s11_bayes_shared_cov', 's11_ds_shared_cov', 'peer_on_rf_opaque', 'peer_off_rf_opaque')
    for k in required_cases:
        if k not in cases:
            abl.append(f'missing required ablation case {k}')
    for k, case in cases.items():
        if not isinstance(case, dict) or case.get('safety_failed_worlds') != 0 or case.get('overflow_worlds') != 0:
            abl.append(f'failure or missing counts in case {k}')

    ok = all(not v for v in reasons.values())
    return {
        'scenario_frozen': ok,
        'status': 'SCENARIO_FROZEN' if ok else 'NOT_FROZEN',
        'gates': {k: {'pass': not bool(v), 'blocking_reasons': v} for k, v in reasons.items()},
        'phase2_matd3_masac_training_permitted': ok,
        'tag_or_push_performed': False,
        'method': 'fail-closed evidence contract; no automatic Git mutation',
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--mac', type=Path, required=True)
    p.add_argument('--endurance', type=Path, required=True)
    p.add_argument('--ablation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    res = evaluate(*(json.loads(x.read_text()) for x in (a.mac,a.endurance,a.ablation)))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(res, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if res['scenario_frozen'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
