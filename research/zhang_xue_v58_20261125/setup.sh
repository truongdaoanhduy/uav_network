#!/usr/bin/env bash
# Repeatable environment bootstrap; no Kaggle credentials or generated data tracked.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$here/../.." && pwd)"
check_only=false
dry_run=false
root_args=()
for option in "$@"; do
  case "$option" in
    --check-only) check_only=true ;;
    --dry-run) dry_run=true ;;
    --cloud|--system-site-packages) root_args+=("$option") ;;
    *) echo "Usage: bash setup.sh [--check-only|--dry-run] [--cloud] [--system-site-packages]" >&2; exit 2 ;;
  esac
done
if "$dry_run"; then
  printf 'Root bootstrap: bash %q/setup.sh' "$repo"
  printf . %q. "${root_args[@]}"
  printf '\nPinned UavNetSim revision: '
  cat "$repo/scripts/upstream-revision.txt"
  printf '\nGenerated scenarios: constructed deterministically from seed, no dataset download.\n'
  exit 0
fi
if ! "$check_only"; then
  bash "$repo/setup.sh" "${root_args[@]}"
fi
python_bin="$repo/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin="${PYTHON:-python3}"
  if ! "$check_only"; then
    echo "Missing installed Python environment after root setup" >&2
    exit 2
  fi
fi
PYTHONPATH="$here/src:$repo${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -c '
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.configuration.loader import load_scenario
import numpy
print("Zhang-Xue package imports OK; benchmark map width:",load_scenario().map_size_m)
'
printf 'Ready: PYTHONPATH=%s/src %s scripts/run_scenario.py --config configs/cpu_smoke.yaml --steps 4\n' "$here" "$python_bin"
