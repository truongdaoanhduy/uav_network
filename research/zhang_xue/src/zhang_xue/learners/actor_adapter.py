"""Bridge the new 7-radio Dec-POMDP environment to retained MATD3/MASAC.

The earlier learner assumed obs[..., 9] is the alive mask and the last
(num_agents+1) observation values represent legal destination actions.
Our experimental sensor map instead returns 514 features plus an external
boolean route mask. We DO NOT silently reinterpret the map's feature #9.
The original distance scalar is preserved as an extra feature.

Original policy motor branch has XY + TX head. This scenario has fixed TX
power, so TX action is intentionally ignored (not silently used as watts).
"""
from __future__ import annotations

import numpy as np

ACTOR_N=6
RAW_DIM=514
DISC_DIM=7
ACTOR_POLICY_DIM=RAW_DIM+1+DISC_DIM
ALIVE_FEATURE_INDEX=9
SOURCE_SOC_INDEX=4


def encode_actor_input(observation, route_mask):
    """Return policy obs shape (...,6,522), without reading ground truth."""
    obs=np.asarray(observation,dtype=np.float32)
    mask=np.asarray(route_mask)
    if obs.shape[-2:] != (ACTOR_N,RAW_DIM):
        raise ValueError("Expected local 7-radio actor observations (..., 6, 514)")
    if mask.shape != obs.shape[:-1]+(DISC_DIM,):
        raise ValueError("Expected separate route mask (..., 6, 7)")
    if not np.isfinite(obs).all():
        raise ValueError("Actor observation must be finite")
    if not np.all(np.isin(mask,[0,1])) or not np.all(mask[...,0]==1):
        raise ValueError("Route mask must be boolean with silent action legal")
    encoded=obs.copy()
    saved_gcs_distance=encoded[...,ALIVE_FEATURE_INDEX:ALIVE_FEATURE_INDEX+1].copy()
    encoded[...,ALIVE_FEATURE_INDEX]=(obs[...,SOURCE_SOC_INDEX]>0).astype(np.float32)
    return np.concatenate((encoded,saved_gcs_distance,mask.astype(np.float32)),axis=-1)


def make_critic_state(encoded_obs):
    """Centralized critic sees joint local policy inputs, not truth/oracle."""
    enc=np.asarray(encoded_obs,dtype=np.float32)
    if enc.shape[-2:] != (ACTOR_N,ACTOR_POLICY_DIM):
        raise ValueError("Expected encoded actors (...,6,522)")
    if not np.isfinite(enc).all():
        raise ValueError("Nonfinite encoded observation")
    return enc.reshape(enc.shape[:-2]+(ACTOR_N*ACTOR_POLICY_DIM,)).copy()


def decode_policy_action(continuous,destination,route_mask,*,v_max_mps):
    """Turn XY network controls into bounded m/s; never use RF head.

    The policy's arbitrary action outside [-1,1] is clipped then normalized
    onto a unit disk (to obey vector speed bound, not just per-axis bounds).
    Illegal network destinations fail closed rather than being silently sent.
    """
    controls=np.asarray(continuous,dtype=np.float64)
    dest=np.asarray(destination)
    mask=np.asarray(route_mask)
    if controls.shape!=(ACTOR_N,3) or dest.shape!=(ACTOR_N,) or mask.shape!=(ACTOR_N,DISC_DIM):
        raise ValueError("Expected continuous (6,3), destination (6,), route mask (6,7)")
    if not np.isfinite(controls).all() or not np.isfinite(v_max_mps) or v_max_mps<=0:
        raise ValueError("Controls and maximum speed must be finite")
    if not np.issubdtype(dest.dtype,np.integer) or np.any((dest<0)|(dest>=DISC_DIM)):
        raise ValueError("Destination must be integer 0..6")
    if not np.all(np.isin(mask,[0,1])) or not np.all(mask[:,0]==1):
        raise ValueError("Route mask must be binary and include silent")
    if np.any(mask[np.arange(ACTOR_N),dest]==0):
        raise ValueError("Policy chose an unavailable destination")
    xy=np.clip(controls[:,:2],-1,1)
    speeds=np.linalg.norm(xy,axis=1,keepdims=True)
    xy=xy/np.maximum(speeds,1)
    return (xy*v_max_mps).astype(np.float64),dest.astype(np.int64,copy=True)
