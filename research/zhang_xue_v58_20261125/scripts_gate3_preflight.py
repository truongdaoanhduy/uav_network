"""CPU-only Gate 3 readiness check. Exit status 2 deliberately blocks full CUDA training."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from zhang_xue_v54_7node_warp import audit_gate3_warp_parity_contract
s=audit_gate3_warp_parity_contract()
print(json.dumps(s,ensure_ascii=False,indent=2))
raise SystemExit(0 if s['gate3_ready'] else 2)
