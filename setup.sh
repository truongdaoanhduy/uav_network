#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
dry_run=false
check_only=false
gpu=false
while (( $# )); do
  case "$1" in
    --dry-run) dry_run=true ;;
    --check-only) check_only=true ;;
    --gpu) gpu=true ;;
    *) echo "Usage: bash setup.sh [--dry-run] [--check-only] [--gpu]" >&2; exit 2 ;;
  esac
  shift
done
if "$dry_run"; then
  echo "Create local .venv and install algorithm core plus seven-radio scenario"
  echo "Map and target generation use deterministic seeded sampling (no external dataset)"
  if "$gpu"; then echo "Install optional Warp 1.18.0; NVIDIA driver required separately"; fi
  exit 0
fi
python_bin="python3"
if [[ -n "${PYTHON:-}" ]]; then
  python_bin="$PYTHON"
fi
if ! "$check_only"; then
  "$python_bin" -m venv "$root/.venv"
  python_bin="$root/.venv/bin/python"
  "$python_bin" -m pip install -e "$root[dev]"
  "$python_bin" -m pip install -e "$root/research/zhang_xue"
  if "$gpu"; then "$python_bin" -m pip install warp-lang==1.18.0; fi
elif [[ -x "$root/.venv/bin/python" ]]; then
  python_bin="$root/.venv/bin/python"
fi
PYTHONPATH="$root:$root/research/zhang_xue/src" "$python_bin" -c '
import torch,numpy,yaml
from uav_network.algorithms.matd3 import MATD3
from uav_network.algorithms.masac import MASAC
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
print("Imports OK; no training or CUDA execution claimed")
'
