"""V79 behavioural gate: real Warp CPU/CUDA masked reset isolation and terminal rollback.

Independent of Torch and Kaggle credentials. Exit 0 only if both assertions pass.
Use CPU first and then CUDA:0; a PASS on CPU does NOT imply CUDA parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]


def _snapshot(w):
    n = w.network
    b = w.buffers
    return {
        "now": n.state.now.numpy().copy(),
        "mission_epoch_us": n.state.mission_epoch_us.numpy().copy(),
        "episode_id": b["episode_id"].numpy().copy(),
        "episode_step": b["episode_step"].numpy().copy(),
        "pos": b["positions"].numpy().copy(),
        "battery": b["battery"].numpy().copy(),
        "truth": b["truth"].numpy().copy(),
        "ledger_pos": n.state.ledger_pos.numpy().copy(),
        "ledger_gen": n.state.ledger_gen.numpy().copy(),
        "reward": b["reward"].numpy().copy(),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True, type=Path)
    a=p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result={"device": a.device, "gate": "V79_SELECTIVE_RESET_AND_ABORT", "status": "RUNNING", "checks": {}}
    try:
        import warp as wp
        from zhang_xue.gpu.mission_v79 import WarpParallelMissionV79, GPUWorldV79Params
        from zhang_xue.gpu.kernels.mission_v79_kernels import _move_v79
        cfg=GPUWorldV79Params.from_yaml(ROOT/"research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml", envs=4, device=a.device)
        world=WarpParallelMissionV79(cfg)
        n=world.network
        b=world.buffers
        acts=wp.zeros((4,6,2),dtype=wp.float32,device=a.device)
        dsts=wp.zeros((4,6),dtype=wp.int32,device=a.device)
        world.step(acts,dsts)
        before=_snapshot(world)
        mask=np.asarray([0,1,0,0],dtype=np.uint8)
        wp.copy(b["done"],wp.array(mask,dtype=wp.uint8,device=a.device))
        world.auto_reset_gpu()
        after=_snapshot(world)
        for name in before:
            if not np.array_equal(before[name][[0,2,3]],after[name][[0,2,3]]):
                raise AssertionError("non-selected world modified during reset: "+name)
        assert int(after["now"][1])==600_000, after["now"]
        assert int(after["mission_epoch_us"][1])==600_000, after["mission_epoch_us"]
        assert int(after["episode_id"][1])==int(before["episode_id"][1])+1
        assert int(after["truth"][1].sum())==15
        assert np.all(b["done"].numpy()==0)
        result["checks"]["masked_reset_preserves_other_worlds"]="PASS"
        result["checks"]["mission_clock_epoch"]="PASS"
        result["checks"]["new_episode_new_targets"]="PASS"

        # Synthetic adjacent UAVs create a guaranteed pairwise unsafe proposal.
        # Post-check MUST abort commit, not merely report failed while moving.
        xy=after["pos"].copy()
        xy[0,0,:2]=[500.0,500.0]
        xy[0,1,:2]=[515.0,500.0]
        wp.copy(b["positions"],wp.array(xy,dtype=wp.float32,device=a.device))
        before_pos=b["positions"].numpy().copy()
        wp.launch(_move_v79,dim=4,inputs=[
            b["positions"],b["velocities"],b["candidates"],b["v_safe"],b["pos_xy"],
            b["battery"],world.build_wp,len(world.buildings),n.state,b["failed"],b["safety_reason"],
            b["unsafe_mask"],b["intervention"],float(cfg.v_max_mps),float(cfg.a_max_mps2),
            float(cfg.uav_safe_distance_m),float(world.pa_dc_w),
        ],device=a.device)
        assert int(b["failed"].numpy()[0])==1
        assert np.array_equal(before_pos[0],b["positions"].numpy()[0]),"unsafe proposal committed"
        result["checks"]["unsafe_postcheck_no_position_commit"]="PASS"
        result["status"]="PASS"
    except ImportError as ex:
        result["status"]="BLOCKED_DEPENDENCY"
        result["error"]=str(ex)
    except Exception as ex:
        result["status"]="FAIL"
        result["error"]=str(ex)
        result["traceback"]=traceback.format_exc()
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0 if result["status"]=="PASS" else (3 if result["status"].startswith("BLOCKED") else 1)

if __name__=="__main__":
    sys.exit(main())
