# Progress — docs/implementation-plan.md

Baseline: the requested directory contains README.md and a clean Git checkout.
The old project is read only; its network action mapping is retained.

Decisions: work directly in the user's requested directory on a feature branch.
Execute stages sequentially under the user's request to build the new project.
Defaults and scope boundaries are recorded in docs/design.md.

Pre-flight: environment produces Action/ReportStore consumed by communication;
MASAC consumes local observations plus separate centralized mission state.
The same signatures and destination conventions apply across all three stages.

Task 1 complete: environment tests ran RED (missing package) then GREEN,
10/10 checks passed on the user's Python 3.13 installation.

Task 2 complete: backend tests ran RED (missing backend) then GREEN.
All 19 environment/network tests passed using the real pinned UavNetSim.
Evidence includes direct GCS delivery, two-hop custody, four-fragment
reassembly, original TTL, failed-PHY energy and same-seed network metrics.

Task 3 complete: learner/experiment tests ran RED (missing modules) then GREEN.
26/26 checks passed. Real-network smoke completed 2 episodes / 80 steps /
73 updates, then CPU evaluation saved metrics and a 1200×1200 PNG.

Independent whole-project review: no Critical findings; two Important defects
reproduced with real UavNetSim. Added failing checks for receiver-buffer
contention and oldest-report eligibility. Fixed the oldest-report case by
selecting the first eligible report for the policy's chosen destination.
Fixed contention by discarding rejected reassembly while retaining sender
custody, so a later action can retransmit. Both require a green full suite.

Final fix verification: both regression checks GREEN; full suite 30/30 GREEN.
The repeated real-network smoke ran 2 episodes / 80 steps / 73 Adam updates.
Both runs' episode logs and all 206 checkpoint tensors match exactly at seed 44.
CPU evaluation wrote metrics and PNG; setup.sh was rerun successfully.
Git exclusions verified for the external runtime, venv and output checkpoints.

Minor finding deferred: custom source-directory upstream_path is trusted;
directory-based loading does not independently check HEAD against the pin.
The default downloaded checkout and installed fallback were verified at
04daafb815eb377409b40b285574eeb62b9a8d58. Installed fallback checks the pin.

## Extension: battery, MATD3 and processes

Plan: `battery-parallel-plan.md`. Work branch: `feature/battery-matd3-vector`.
Battery tests first failed because the energy configuration was missing.
Added propulsion budgets, safe stationary exhaustion, battery observation/state,
failure termination and native PHY frame affordability. GCS ACK energy is
separate from UAV batteries. Destination masks expose live in-range candidates.
Real peer channel tests confirm obstacle LOS→NLOS attenuation and a configurable
extra 10 dB loss gives exactly 0.1× gain while LOS links remain unchanged.

Stage 1: all 40 tests passed on the user's machine. Integer-valued battery config
initially produced integer arrays; the regression now confirms floating budgets.
GPU/parallel library research is captured in `gpu-acceleration-research.md`.
