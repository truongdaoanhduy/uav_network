"""V121 evidence overlay: preserve 72 historical records; append reproducible new findings.

This ledger deliberately never rewrites original root causes or historical PASS results.
A bounded GPU PASS changes a gate's evidence, not its scientific validity class.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/v121_scenario_audit"
OLD = ROOT / "outputs/v120_conflict_resolution/V120_ALL_72_CONFLICTS_RESOLUTION_MATRIX.json"

EVIDENCE = {
    "C04": ("LIMITED_CALIBRATION_OPEN", "V120 correlated Markov 3 seeds x300; V121 command units corrected, rerun pending; independent assumption not measured on physical sensor", "5+ seeds x900 + error rates including missed detections with realistic temporal dependence"),
    "C05": ("LIMITED_CALIBRATION_OPEN", "V120 threshold 0.90/0.95/0.99 trade-off; old V120 calibrated motor commands were mis-scaled", "Recheck paired real-length GCS outcomes with corrected units and uncertainty"),
    "C07": ("SURROGATE_SCOPE_LOCK_ONLY", "CPU same-deadline DIFS/backoff and HELLO frame tests V121; not upstream SimPy packet-event parity", "Do not claim UavNetSim implementation parity without upstream trace"),
    "C08": ("CONSTRUCTIVE_WITNESS_LIMITED", "V121 constructive 10-seed 15/15 oracle-like witness reported historically; not RL or formal global feasibility proof", "Full reproduction and multi-policy feasibility with obstacle/network controls"),
    "C11": ("CANONICAL_INTERFACE_DEFINED", "XY continuous actor only; 0.1W TX, pure-GCS deterministic comm; no hybrid learned channel", "Phase 2 verify actor adapter and objective against MATD3/MASAC original papers"),
    "C14": ("PASS_UNINTERRUPTED_GPU_BOUNDED", "Two post-shield true no-reset 4096x900 CUDA workloads PASS, Base and RF; outputs/v121_scenario_audit/cuda_uninterrupted_*_post_tie.json", "Multi-seed action/sensor/network sweeps and failure modes; no universal guarantee"),
    "C16": ("VERIFIED_DS_ONLY", "V121 direct schema/loader rejects Bayesian runtime; active YAML hashes retained", "None for active runtime; historical unused Bayes helpers remain documented"),
    "C20": ("PURE_GCS_BOUNDARY_LOCK", "PEER/MULE disabled, rule-GCS-only; no claim that PEER is universally harmful", "If future research needs PEER run matched non-oracle cost/utility control"),
    "C23": ("MASS_MODEL_NOT_SENSOR_CALIBRATED", "mf=me=.8 is a reliability parameter, not derived from Pd=.85, Pf=.03 by sensor physics", "Calibrate uncertainty or bound FP under correlated noise before external-validity claim"),
    "C24": ("DEFINITIONS_FIXED", "NO_APP_SEND suppresses application DATA, not HELLO/ACK control-plane airtime", "Clarify control-plane airtime vs packet-size parity"),
    "C26": ("LEARNING_BOUNDARY_LOCK", "Baseline learns XY only; event-driven communication rules not learned", "None in Phase1; Phase2 learner validation"),
    "C27": ("LIMITED_NON_ORACLE_SENTINEL_PASS", "V120 out-of-FOV truth perturbation 3 seeds x5 steps checks obs/state/mask", "Long-trace randomized perturbation and actor replay taint"),
    "C31": ("NOT_FROZEN", "V121 closes specified engineering no-reset CUDA gate; independent scientific assumptions still open", "Sign-off on explicit model scope plus full remaining science gates"),
    "C33": ("SURROGATE_SCOPE_LOCK_ONLY", "HELLO bytes and equal-time CPU contention corrected; Warp/SImPy upstream event parity unverified", "Trace-level cross-check if making UavNetSim-equivalence claims"),
    "C34": ("PURE_GCS_CANONICAL_DECISION", "V119 RF PEER decrease observed under one heuristic; keep PEER=false, MULE=false", "Never generalize single controller result"),
    "C38": ("SHIELD_MULTISTEP_PASS_BOUNDED", "V121 5 worlds x180=900 world-steps after deterministic tie-policy: max 3.86e-6m/s, 0 divergent; before max 2.87m/s", "Full belief/network/reward GPU/CPU state parity not claimed"),
    "C39": ("EVER_VS_ACTIVE_DEFINED", "V120 latch remains ever-confirmed while active belief can drop", "Show both metrics; do not label ever-latch as current detection quality"),
    "C41": ("SUM_ADAPTATION_LIMITED", "SUM source counts != Zhang MAX source fusion; repeated same-UAV evidence still correlated", "Design ablation with robust repeated-evidence variants; no formula attribution to Zhang"),
    "C55": ("PASS_UNINTERRUPTED_GPU_BOUNDED", "V121 fresh post-tie Base+RF 4096x900 no reset PASS", "Failure-rate bounds for other seeds/adversarial controls"),
    "C65": ("SURROGATE_SCOPE_LOCK_ONLY", "Compressed frame 70+8k B and Warp MAC differ from upstream SimPy", "Event-level parity required only for strict emulation claim"),
    "C66": ("TEMPORAL_DEPENDENCE_OPEN", "V120 rho sweep shows false confirmation risk from self-repeated evidence", "Correlated multi-seed 900-step calibration + explicit scope"),
    "C67": ("PASS_UNINTERRUPTED_GPU_BOUNDED", "V121 post-shield 4096x900 Base and RF PASS without resets", "Not full MARL robustness proof"),
    "C68": ("SHIELD_ABI_AND_TIE_PASS_BOUNDED", "Nominal action disk normalization V119 and common-prestate V121 shield differential PASS", "Full-state parity remains open"),
    "C71": ("REPRO_TRACE_IMPROVED_PARTIAL", "V121 adds JSONL CUDA, shield counterexample and fail logs", "Include complete deterministic natural-mission controller trace and paired evaluation"),
    "C72": ("RESET_LIMITATION_PARTLY_CLOSED", "V119 reset-mode limitation explicitly repaired by V121 no-reset 4096x900 Base/RF workload", "Broader non-stationary policy stress and termination semantics"),
}
# Corrected V121 CPU calibration: 3 seeds x300, 45 DS + 45 RF runs.
# These overlay newer empirical results without rewriting historical V120 data.
EVIDENCE.update({
    "C04": ("OPEN_CORRELATED_FP", "V121 3-seed corrected-action DS SUM FP-ever/world: rho=0:0.333, rho=.5:8.333, rho=.7:13.000; 300 steps", "Calibrate temporal sensor dependence and add >=5 independent seeds x900"),
    "C05": ("LIMITED_THRESHOLD_TRADEOFF", "V121 rho=.7: threshold .90 gives FP-ever13, TP-ever11; threshold .99 gives FP-ever8.333, TP-ever10 (3 seeds x300)", "Predeclared FP/TP tradeoff and longer diverse sensor distributions"),
    "C12": ("MODEL_SENSITIVITY_OPEN", "V121 correct 15m/s units: accel_energy=False mean minimum battery74.12% vs enabled70.50% after300steps; NOT <.01%", "Validate accel motor energy assumptions against measured propulsion data before choosing canonical"),
    "C13": ("RF_ATTENUATION_SYNTHETIC", "V121 PDR base building_loss 0dB=.9613, 12dB=.9409, 20dB=.9240; 3 seeds x300", "Building material/geometry loss experimental or measured scenario calibration"),
    "C23": ("MASS_RELIABILITY_NOT_PHYSICAL", "V121 corrected motion rho=0: MF cap .80 FP-ever0.333; cap .99 FP-ever8.333, three seeds x300", "Report mass design choice and test sensor data reliability outside synthetic world"),
    "C28": ("NETWORK_SUBSTEPS_SMALL_SENSITIVITY", "V121 3-seed corrected-action substeps1 vs2 base PDR .9409 vs .9403; RF .2387 vs .2389", "Document beginning-of-step radio geometry and validate dynamics under fast motion"),
    "C35": ("THRESHOLD_EFFECT_OPEN", "V121 corrected rho=.7 FP-ever 13 at threshold .90 vs8.333 at threshold .99 with TP loss 11 to10", "Select threshold only on separately calibrated reliability target, not reward improvement"),
    "C41": ("SOURCE_FUSION_LIMITED", "V121 corrected-action rho=.7 FP-ever13 both SUM and Zhang MAX, due within-source repeated evidence", "Ablate temporal correlation and independent evidence source before attributing paper behavior"),
    "C66": ("TEMPORAL_CORRELATION_OPEN", "V121 corrected-action 45 DS runs: rho=0/.5/.7 FP-ever .333/8.333/13 for SUM", "Account repeated observations before making confident GCS confirmation claims"),
})

NEW = [
("C73","P1","VALIDATION_BUG","scripts/gpu/v99_endurance_sticky_cuda_gate.py", "V120 command used --worlds although legacy gate only supports --envs; and its unconditional auto-reset invalidates uninterrupted episode claim", "v121_uninterrupted_cuda_gate.py: explicit no-reset; 4096x900 Base+RF PASS", "FIXED_FOR_DEFINED_WORKLOAD", "scripts/gpu/v121_uninterrupted_cuda_gate.py; CUDA Base+RF JSON", "More motion/noise seeds", "Engineering semantics, not a paper formula"),
("C74","P1","UNIT_BUG","scripts/maintenance/v120_*calibration.py", "Sector controller returned normalized length 1 but CPU env.step expects XY velocity in m/s (15 m/s max)", "unit test RED/GREEN; corrected conversion, recalibration repeated", "PATCHED_CALIBRATION_RECHECK", "tests/test_v121_calibration_contracts.py; calibrations/", "Corrected full-length 900 step calibration", "Unit convention: actor_adapter & CPU transition"),
("C75","P1","CPU_MAC_RACE","communication/scheduler.py", "Simultaneous DIFS/backoff expiries were committed sequentially by radio index; hidden terminals could falsely avoid collisions", "Ready-snapshot and exact slot tests GREEN", "FIXED_CPU", "tests/test_v121_network_parity.py", "CPU/Warp and SimPy upstream event trace differential", "IEEE 802.11 DCF concurrency; custom Warp implementation"),
("C76","P1","AIRTIME_ACCOUNTING","communication/common.py; communication/channel.py", "HELLO peer-telemetry bytes not charged on CPU; Warp V89 already charged telemetry", "Seven-node HELLO duration plus frame/ACK regression tests GREEN", "FIXED_CPU", "tests/test_v121_network_parity.py", "Paired MAC per-event validation", "Custom UavNetSim-inspired surrogate"),
("C77","P0","DS_RUNTIME_BYPASS","environment/configuration.py; environment/transition.py", "Legacy CPU config and direct runtime override could enter Bayesian belief despite DS-only active YAML", "Direct config/CLI CPU tests RED/GREEN", "FIXED_CANONICAL", "tests/test_v121_ds_only_runtime.py", "Historic Bayesian standalone functions deliberately kept for noncanonical tests", "Zhang 2024 DS theoretical basis, implementation scope choice"),
("C78","P1","CPU_CUDA_TIE","mobility/flight_safety_energy.py; gpu/kernels/mission_v80_shield.py", "FP32 vs FP64 near equal projection costs produced distinct candidate enumeration, up to 2.87m/s", "Shared finite tolerance 64*FP32 epsilon on cost only; post 900 world-step diff max 3.86e-6m/s", "FIXED_BOUNDED", "tests/test_v121_shield_tie_break.py; shield_post_tie_cuda.json", "More seeds & adversarial geometries", "Numerical determinism; not a new CBF or safety physics"),
("C79","P2","TEST_FIXTURE_SCOPE","tests/test_v121_shield_tie_break.py", "Test invoked geometry-only projection check_peers=0 yet expected CPU full three-pass peer shield result", "Initial suite 6 subfails; corrected test and 4/4 GREEN (9 subtests)", "FIXED_TEST", "pytest_post_tie_full.stdout.log; test_shield_fixture_corrected.log", "Full pytest 640 PASS / 6 SKIP / 152 subtests; no remaining regression gate", "Test API boundary, not literature"),
("C80","P2","STALE_METADATA","docs/superpowers/CURRENT_TASK.md; tests/test_v121_active_metadata.py", "Active project documentation called obsolete Bayes configuration despite DS-only", "Metadata test RED/GREEN and loader rejects incompatible fallback", "FIXED_METADATA", "tests/test_v121_active_metadata.py", "None for current metadata", "Project provenance"),
("C81","P0","SCIENTIFIC_CLAIM","docs/V120_ALL_72_CONFLICTS_ROOT_CAUSE_AND_SOLUTIONS_VI.md", "V120 claimed only two GPU checks needed to freeze despite correlated DS and full CPU/GPU/SimPy parity gaps", "V121 finite workload proves only a bounded engineering gate; status fail-closed", "OPEN_LIMIT_CLAIMS", "V121 report; C04/C07/C23/C38/C41/C66", "Freeze only after explicit scientific scope and remaining validation", "Model validity governance"),
("C82","P1","FEASIBILITY_CLAIM","scripts/maintenance/v120_cpu_rf_substeps_energy_calibration.py", "Straight-line target-aware path ignoring obstacles/routing falsely described as end-to-end mission feasibility", "Relabeled idealized bound; constructive actual non-RL mission witness available for limited seeds", "CORRECTED_CLAIM_BOUNDED", "scripts/maintenance/v121_constructive_mission_witness.py", "Natural non-oracle missions across target configurations", "Zeng 2019 energy model + author-derived geometric scheduling"),
]

def main():
    old=json.loads(OLD.read_text())['conflicts']
    assert len(old)==72 and [r['ID'] for r in old]==[f'C{i:02d}' for i in range(1,73)]
    out=[]
    for item in old:
        row=dict(item)
        row["V121_current_assessment"],row["V121_verified_evidence"],row["V121_remaining_gate"] = EVIDENCE.get(
            row['ID'],("INHERITED_V120_NOT_REVALIDATED","Previous evidence/status preserved; source code tests may include historical cases",row.get("V119_remaining_gate", "Independent evaluation if claimed")))
        row["V121_historical_root_cause_preserved"]=True
        out.append(row)
    for i, severity, typ, path, root, fix, status, evidence, gate, provenance in NEW:
        out.append(dict(ID=i,Class=typ,Severity=severity,Source=path,Root_cause=root,
            Evidence=fix,Resolution_or_proposed_action=fix, Status=status,
            Paper_reference_or_provenance=provenance,
            V121_current_assessment=status,V121_verified_evidence=evidence,
            V121_remaining_gate=gate,V121_historical_root_cause_preserved=False))
    assert [r['ID'] for r in out]==[f'C{i:02d}' for i in range(1,83)]
    OUT.mkdir(parents=True,exist_ok=True)
    jpath=OUT/'V121_CONFLICT_REGISTRY_C01_C82.json'
    jpath.write_text(json.dumps(dict(total=82,source_historical='V120 C01-C72',conflicts=out),ensure_ascii=False,indent=2)+'\n')
    fields=['ID','Severity','Class','Source','Root_cause','V120_status',
            'V121_current_assessment','V121_verified_evidence','V121_remaining_gate',
            'Paper_reference_or_provenance','V121_historical_root_cause_preserved']
    with (OUT/'V121_CONFLICT_REGISTRY_C01_C82.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(out)
    print('WROTE',len(out),'conflicts, old records preserved',jpath)
if __name__=='__main__': main()
