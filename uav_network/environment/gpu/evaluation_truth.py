"""Truth-aligned metrics for the host-only WarpMissionEnv evaluation adapter.

NEVER use this in Actor/critic inputs, admission masks, MAC or on-device
NoOracle report creation: local policies must see only their own evidence.
"""
import numpy as np


def truth_aligned_report_status(targets, report_cells, uav_known, delivered, grid_n, cell_width):
    """Project internal cell-hypothesis REPORT slots to target-indexed metrics.

    Returns (per_uav_truth_seen: [N,T], truth_delivered: [T]). False
    hypotheses have no matching target and therefore never inflate scores.
    """
    targets = np.asarray(targets, dtype=np.float64)
    cells = np.asarray(report_cells, dtype=np.int64)
    known = np.asarray(uav_known, dtype=bool)
    received = np.asarray(delivered, dtype=bool)
    if (targets.ndim!=2 or targets.shape[1]!=2 or cells.ndim!=1
            or known.ndim!=2 or known.shape[1]!=len(cells)
            or received.shape!=cells.shape or not isinstance(grid_n,(int,np.integer))
            or grid_n<1 or not np.isfinite(cell_width) or cell_width<=0
            or not np.isfinite(targets).all()):
        raise ValueError('Invalid NoOracle truth-scoring tensor shapes')
    x=np.clip((targets[:,0]/cell_width).astype(np.int64),0,grid_n-1)
    y=np.clip((targets[:,1]/cell_width).astype(np.int64),0,grid_n-1)
    target_cell=y*grid_n+x
    matches=(cells[:,None]>=0)&(cells[:,None]==target_cell[None,:])
    found=np.any(known[:,:,None]&matches[None,:,:],axis=1)
    delivered_truth=np.any(received[:,None]&matches,axis=0)
    return found,delivered_truth
