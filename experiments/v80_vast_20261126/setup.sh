#!/usr/bin/env bash
# Single entry point for ALL source packages, data, tests and optional Warp preflight.
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
cd "$root"
install=false; check_only=false; prepare_data=false; test_all=false; cpu_gate=false; cpu_gate_datamule=false; cpu_gate_v67=false; qa=false
gpu_gate=false; gpu_packet_gate=false; gpu=false; dry_run=false
seed=44; episodes=4
while (( $# )); do
  case "$1" in
    --install) install=true ;;
    --gpu) gpu=true; install=true ;;
    --check-only) check_only=true ;;
    --prepare-data) prepare_data=true ;;
    --test) test_all=true ;;
    --cpu-gate) cpu_gate=true ;;
    --cpu-gate-datamule) cpu_gate_datamule=true ;;
    --cpu-gate-v67) cpu_gate_v67=true ;;
    --gpu-gate) gpu_gate=true ;;
    --gpu-packet-gate) gpu_packet_gate=true ;;
    --qa) qa=true ;;
    --dry-run) dry_run=true ;;
    --seed) shift; seed="${1:?missing seed}" ;;
    --episodes) shift; episodes="${1:?missing episodes}" ;;
    -h|--help)
      echo 'Usage: bash setup.sh [--install] [--gpu] [--prepare-data] [--check-only] [--test] [--cpu-gate] [--cpu-gate-datamule] [--cpu-gate-v67] [--gpu-gate] [--gpu-packet-gate] [--qa] [--dry-run] [--seed N] [--episodes N]'
      exit 0 ;;
    *) echo "Unknown setup option: $1" >&2; exit 2 ;;
  esac
  shift
done
if ! "$install" && ! "$check_only" && ! "$prepare_data" && ! "$test_all" && ! "$cpu_gate" && ! "$cpu_gate_datamule" && ! "$cpu_gate_v67" && ! "$gpu_gate" && ! "$gpu_packet_gate" && ! "$qa"; then
  install=true; check_only=true; prepare_data=true
fi
if "$dry_run"; then
  echo 'PLAN: algorithm core + seven-radio package install (only when requested), local synthetic data, optional tests and CPU gates.'
  echo 'PLAN: no external dataset required; optional HTTPS sources require SHA-256; outputs ignored by Git.'
  echo "PLAN: install=$install gpu=$gpu check=$check_only data=$prepare_data tests=$test_all cpu_gate=$cpu_gate cpu_gate_datamule=$cpu_gate_datamule cpu_gate_v67=$cpu_gate_v67 gpu_gate=$gpu_gate gpu_packet_gate=$gpu_packet_gate qa=$qa seed=$seed episodes=$episodes"
  exit 0
fi
python_bin="${PYTHON:-python3}"
if "$install"; then
  "$python_bin" -m venv "$root/.venv"
  python_bin="$root/.venv/bin/python"
  "$python_bin" -m pip install -e '.[dev]'
  "$python_bin" -m pip install -e './research/zhang_xue'
  if "$gpu"; then "$python_bin" -m pip install 'warp-lang==1.18.0'; fi
elif [[ -x "$root/.venv/bin/python" ]] && [[ -z "${PYTHON:-}" ]]; then
  python_bin="$root/.venv/bin/python"
fi
export PYTHONPATH="$root:$root/research/zhang_xue/src"
if "$check_only"; then
  "$python_bin" - <<'PY'
from uav_network.algorithms.matd3 import MATD3
from uav_network.algorithms.masac import MASAC
from uav_network.algorithms.configuration import load_learner_config
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from scripts.data.prepare_data import validate_manifest
print('Repository imports and setup manifest tools: PASS (not a GPU test)')
PY
  "$python_bin" scripts/maintenance/check_project_policy.py --scope full
fi
if "$prepare_data"; then
  "$python_bin" scripts/data/prepare_data.py --seed "$seed" --episodes "$episodes"
fi
if "$test_all"; then
  "$python_bin" -m pytest -q tests/ research/zhang_xue/tests/
fi
if "$cpu_gate"; then
  for algo in matd3 masac; do
    "$python_bin" research/zhang_xue/scripts/v61_cpu_actor_learning_gate.py \
      --algorithm "$algo" --seed "$seed" --transitions 4 \
      --learner-config "configs/learners/${algo}_cpu_gate.yaml" \
      --scenario-config research/zhang_xue/configs/scenario_7radio.yaml
  done
fi
if "$cpu_gate_datamule"; then
  for algo in matd3 masac; do
    "$python_bin" research/zhang_xue/scripts/v61_cpu_actor_learning_gate.py \
      --algorithm "$algo" --seed "$seed" --transitions 4 \
      --learner-config "configs/learners/${algo}_cpu_gate.yaml" \
      --scenario-config research/zhang_xue/configs/scenario_v66_datamule_1km.yaml
  done
fi
if "$cpu_gate_v67"; then
  for algo in matd3 masac; do
    "$python_bin" research/zhang_xue/scripts/v61_cpu_actor_learning_gate.py \
      --algorithm "$algo" --seed "$seed" --transitions 4 \
      --learner-config "configs/learners/${algo}_cpu_gate.yaml" \
      --scenario-config research/zhang_xue/configs/scenario_v67_datamule_1km.yaml
  done
fi
if "$gpu_gate"; then
  # Fail closed until a Q1-Clean 2D + compact-3 GPU runtime is wired and parity tested.
  # The older v61 7-radio Warp smoke is a different experiment.
  "$python_bin" scripts/maintenance/q1_gpu_preflight.py
fi

if "$gpu_packet_gate"; then
  "$python_bin" scripts/gpu/warp_packet_cuda_gate.py --envs "${WARP_GATE_ENVS:-4}" --device "${WARP_GATE_DEVICE:-cuda:0}"
fi
if "$qa"; then
  "$python_bin" scripts/maintenance/run_qa_bundle.py --seed "$seed"
fi
