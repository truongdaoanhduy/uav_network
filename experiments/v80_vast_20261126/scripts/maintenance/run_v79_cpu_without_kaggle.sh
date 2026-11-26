#!/usr/bin/env bash
# Runs the actual V79 Warp CPU implementation (no Kaggle, no GPU or tokens).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${2:-quick}" # quick | medium | stress
DEST="${1:-$ROOT/outputs/v79_cpu_local}"
# Never overwrite old results: a repeated invocation gets an isolated run directory.
if [[ -d "$DEST" && -n "$(find "$DEST" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  DEST="$DEST/run_$(date -u +%Y%m%dT%H%M%S)_$$"
fi
mkdir -p "$DEST"
echo "CPU_LOG_DIR=$DEST"
export PYTHONPATH="$ROOT:$ROOT/research/zhang_xue/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
cd "$ROOT"
"$PYTHON_BIN" -c 'import numpy, torch, warp; print("WARP_CPU_RUNTIME",warp.__version__,"PYTORCH",torch.__version__)' > "$DEST/runtime.log" 2>&1
"$PYTHON_BIN" scripts/gpu/v79_selective_reset_safety_gate.py --device cpu --output "$DEST/selective_reset.json" > "$DEST/selective_reset.log" 2>&1
run_algo() {
  local algo="$1" n="$2" steps="$3" horizon="$4" batch="$5"
  local name="${algo}_e${n}_h${horizon}_steps${steps}"
  "$PYTHON_BIN" scripts/gpu/run_v79_real_rollout.py \
    --algorithm "$algo" --envs "$n" --steps "$steps" --horizon "$horizon" \
    --batch-size "$batch" --hidden 32 --world-device cpu --learner-device cpu \
    --allow-experimental-physics --output "$DEST/$name.json" > "$DEST/$name.log" 2>&1
}
run_algo matd3 64 10 2 64
run_algo masac 64 10 2 64
case "$MODE" in
  quick) ;;
  medium|stress) run_algo matd3 256 6 2 256; run_algo masac 256 6 2 256 ;;
  *) echo "Invalid mode: $MODE (quick|medium|stress)" >&2; exit 2 ;;
esac
if [[ "$MODE" == stress ]]; then
  run_algo matd3 64 60 30 128
  run_algo masac 64 60 30 128
fi
"$PYTHON_BIN" - "$DEST" <<'PY'
from pathlib import Path
import json,sys
folder=Path(sys.argv[1])
rows={}
for p in sorted(folder.glob('*.json')):
    d=json.loads(p.read_text())
    rows[p.name]={k:d.get(k) for k in ('status','updates','world_steps','episodes_ended_total','elapsed_s','device','error')}
    if p.name=='selective_reset.json':
        assert d.get('status')=='PASS',d
    else:
        assert d.get('status')=='V79_CPU_ROLLOUT_PASS_VERIFY_PARITY_AND_CONVERGENCE',d
summary={'overall':'PASS_WARP_CPU_ONLY','gpu_tested':False,'cpu_warp_parity_proven':False,
         'training_converged':False,'runs':rows}
(folder/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
PY
