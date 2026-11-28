#!/usr/bin/env python3
"""Fail-closed Phase-1 scenario acceptance gate. Does not auto-authorize RL training.

A PASS from pytest/CUDA is not a peer-reviewed proof of DS sensor calibration,
MAC packet/event equivalence, or absence of oracle observation. This audit
reports bounded engineering evidence separately from scientific acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
YAML = ROOT / 'research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml'
YAML_RF = ROOT / 'research/zhang_xue/configs/scenario_v105_ds_rf_stress_1km.yaml'
EXPECTED = {
    'base': '3fcfe434b35c29310835c2aa99f982a96c4af58375dc5e916e1e4d017a832fc5',
    'rf': '3ee084e793716baa9ea83d842af0b51345690fdb8156ab0e5c527739ad033b76',
}
BLOCKERS = {
    'G01_DS_TEMPORAL_DEPENDENCE': 'Cooldown CPU/Warp is an opt-in heuristic; sensor temporal rho and correlated end-to-end TP/FP not calibrated',
    'G02_SENSOR_QUALITY': 'Pd=0.85 Pf=0.03 and off-nadir sensor geometry are assumed, not calibrated with measurements',
    'G03_RF_PROPAGATION': '12 dB/obstacle and RF noise are scenario coefficients, not validated material/link measurements',
    'G04_MAC_UPSTREAM': 'Warp CSMA/CA surrogate not validated against matching SimPy UavNetSim-v1 event traces',
    'G05_CPU_CUDA_WHOLE_WORLD': 'No matched multi-step moving CPU/GPU comparison of DS, GCS delivery, queue, energy, reward & observations',
    'G06_NON_ORACLE_MISSIONS': 'Need reproducible non-oracle full 900-step missions across independent target maps and seeds',
    'G07_SUCCESS_SEMANTICS': 'Decide and test ever-confirmed vs currently confirmed success under contradictory DS evidence',
    'G08_TRAINING_POLICY_CONTRACT': 'Baseline canonical 2D action contract needs final Phase1 acceptance before standard MATD3/MASAC',
}


def assess(evidence: dict | None = None, *, yaml_paths: dict[str, Path] | None = None) -> dict:
    paths = yaml_paths or {'base': YAML, 'rf': YAML_RF}
    hashes = {key: (hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None)
              for key, path in paths.items()}
    mismatch = [key for key, want in EXPECTED.items() if hashes.get(key) != want]
    # A user-provided checklist cannot cryptographically prove science: evidence
    # is recorded without promoting the status to FROZEN automatically.
    observed = evidence or {}
    accepted = {k: bool(observed.get(k, {}).get('independent_review_accepted', False))
                for k in BLOCKERS}
    missing = [k for k in BLOCKERS if not accepted[k]]
    can_request_manual_review = not mismatch and not missing
    return {
        'gate': 'V124_PHASE1_FAIL_CLOSED_FREEZE_REVIEW',
        'scenario_frozen': False, 'phase2_baseline_started': False,
        'canonical_act_dim': 2,
        'yaml_sha256': hashes, 'yaml_mismatch': mismatch,
        'scientific_blockers': [{ 'id': k, 'reason': reason,
                                 'accepted_in_supplied_evidence': accepted[k] }
                                for k, reason in BLOCKERS.items()],
        'outstanding': missing, 'can_request_manual_review': can_request_manual_review,
        'status': ('REVIEW_REQUIRED_NOT_FROZEN' if can_request_manual_review else
                   'BLOCKED_NOT_FROZEN'),
        'claim_limit': ('This tool NEVER promotes SCENARIO_FROZEN to true. '
                        'Manual reproducible evidence review and independent validity assessment are required.'),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    evidence = json.loads(args.evidence.read_text()) if args.evidence else None
    report = assess(evidence)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(report['status'], 'missing', len(report['outstanding']), 'hash_mismatch', report['yaml_mismatch'])
    return 2 if report['status'] == 'BLOCKED_NOT_FROZEN' else 1

if __name__ == '__main__':
    raise SystemExit(main())
