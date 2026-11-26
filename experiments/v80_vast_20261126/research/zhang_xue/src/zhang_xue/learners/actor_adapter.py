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
    if controls.ndim != 2 or controls.shape[1] != 3 or not np.isfinite(controls).all():
        raise ValueError("Expected finite continuous controls shape (N, 3)")
    return decode_policy_action_2d(controls[:, :2], destination, route_mask, v_max_mps=v_max_mps)


def decode_tx_power_w(continuous, *, p_min_w: float = 0.05, p_max_w: float = 0.50) -> np.ndarray:
    """Map policy continuous RF head controls[:, 2] in [-1, 1] to per-UAV TX power in Watts."""
    controls = np.asarray(continuous, dtype=np.float64)
    if controls.ndim != 2 or controls.shape[0] < 1 or controls.shape[1] != 3 or not np.isfinite(controls).all():
        raise ValueError("Expected finite continuous controls shape (N, 3)")
    if not (0.0 < p_min_w <= p_max_w):
        raise ValueError("Require 0 < p_min_w <= p_max_w")
    a_rf = np.clip(controls[:, 2], -1.0, 1.0)
    return (p_min_w + 0.5 * (a_rf + 1.0) * (p_max_w - p_min_w)).astype(np.float64)


def step_from_policy_action(env, policy_action, route_mask, *, power_control: str):
    """Run one real packet/flight step from the complete MARL policy action.

    Exactly one of the learned or fixed RF experiments may be selected; do not
    train a dead 3rd actor dimension while silently keeping TX power fixed.
    """
    if power_control not in ("fixed", "learned"):
        raise ValueError("power_control must be 'fixed' or 'learned'")
    learned = power_control == "learned"
    if bool(env.cfg.enable_tx_power_control) != learned:
        raise ValueError("power_control must match enable_tx_power_control in scenario config")
    action_array = np.asarray(policy_action.continuous)
    if action_array.shape == (env.actor_n, 2):
        if learned:
            raise ValueError("learned RF power requires a third actor dimension")
        velocity, destination = decode_policy_action_2d(
            action_array, policy_action.destination, route_mask,
            v_max_mps=env.cfg.v_max_mps,
        )
    else:
        # Learned RF head uses the environment's fleet size and action-mask width.
        velocity, destination = decode_policy_action(
            action_array, policy_action.destination, route_mask,
            v_max_mps=env.cfg.v_max_mps,
        )
    powers = (
        decode_tx_power_w(
            policy_action.continuous,
            p_min_w=env.cfg.tx_power_min_w,
            p_max_w=env.cfg.tx_power_max_w,
        )
        if learned else None
    )
    return env.step(velocity, destination, tx_power_w=powers)


def encode_actor_input_scalable(observation, route_mask):
    """N-invariant actor encoder: accepts (..., N, 514) + (..., N, D) -> (..., N, 515 + D)."""
    obs = np.asarray(observation, dtype=np.float32)
    mask = np.asarray(route_mask)
    if obs.ndim < 2 or obs.shape[-1] != RAW_DIM:
        raise ValueError("Expected local actor observations shape (..., N, 514)")
    if mask.shape[:-1] != obs.shape[:-1] or mask.shape[-1] < 2:
        raise ValueError("Expected route mask shape (..., N, D) with D >= 2")
    if not np.isfinite(obs).all():
        raise ValueError("Actor observation must be finite")
    if not np.all(np.isin(mask, [0, 1])) or not np.all(mask[..., 0] == 1):
        raise ValueError("Route mask must be binary with silent action legal")
    encoded = obs.copy()
    saved_gcs_distance = encoded[..., ALIVE_FEATURE_INDEX : ALIVE_FEATURE_INDEX + 1].copy()
    encoded[..., ALIVE_FEATURE_INDEX] = (obs[..., SOURCE_SOC_INDEX] > 0).astype(np.float32)
    return np.concatenate((encoded, saved_gcs_distance, mask.astype(np.float32)), axis=-1)


def decode_policy_action_2d(continuous_xy, destination, route_mask, *, v_max_mps: float):
    """v6.3 Q1-Clean 2D velocity decoder (continuous_dim=2, NO redundant RF head) for any N UAVs."""
    controls = np.asarray(continuous_xy, dtype=np.float64)
    dest = np.asarray(destination)
    mask = np.asarray(route_mask)
    if controls.ndim != 2 or controls.shape[1] != 2:
        raise ValueError("Expected continuous_xy shape (N, 2) with no redundant RF head")
    n_uav = controls.shape[0]
    if dest.shape != (n_uav,) or mask.ndim != 2 or mask.shape[0] != n_uav:
        raise ValueError("Mismatched N across continuous_xy, destination, and route_mask")
    disc_dim = mask.shape[1]
    if not np.isfinite(controls).all() or not np.isfinite(v_max_mps) or v_max_mps <= 0.0:
        raise ValueError("Controls and maximum speed must be finite positive")
    if not np.issubdtype(dest.dtype, np.integer) or np.any((dest < 0) | (dest >= disc_dim)):
        raise ValueError(f"Destination must be integer in [0, {disc_dim - 1}]")
    if not np.all(np.isin(mask, [0, 1])) or not np.all(mask[:, 0] == 1):
        raise ValueError("Route mask must be binary and include silent")
    if np.any(mask[np.arange(n_uav), dest] == 0):
        raise ValueError("Policy chose an unavailable destination")
    xy = np.clip(controls, -1.0, 1.0)
    speeds = np.linalg.norm(xy, axis=1, keepdims=True)
    xy = xy / np.maximum(speeds, 1.0)
    return (xy * v_max_mps).astype(np.float64), dest.astype(np.int64, copy=True)


def make_critic_state_scalable(encoded_obs):
    """Joint actor-information state; no access to simulator ground truth."""
    enc = np.asarray(encoded_obs, dtype=np.float32)
    if enc.ndim < 2 or enc.shape[-1] < RAW_DIM + 3 or not np.isfinite(enc).all():
        raise ValueError("Expected finite encoded shape (..., N, policy_obs_dim)")
    return enc.reshape(enc.shape[:-2] + (enc.shape[-2] * enc.shape[-1],)).copy()
