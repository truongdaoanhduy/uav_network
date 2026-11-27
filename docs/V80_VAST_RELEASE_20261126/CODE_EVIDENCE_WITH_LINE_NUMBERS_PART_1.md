# Source excerpts — part 1

## 01 Runner torch/Warp dtype
Evidence: 03 original CUDA error; 09 both algorithms PASS
Source excerpt:
scripts/gpu/run_v80_real_rollout.py:135-151
0135:             state = old.reshape(args.envs, -1)
0136:             action = learner.act_batch(old, deterministic=False, return_tensors=True)
0137:             act_cont = action.continuous.to(args.world_device).contiguous()
0138:             act_dest = action.destination.to(device=args.world_device, dtype=torch.int32).contiguous()
0139:             world.step(wp.from_torch(act_cont, dtype=wp.float32), wp.from_torch(act_dest, dtype=wp.int32))
0140:             nextobs = encoded_obs()
0141:             nextstate = nextobs.reshape(args.envs, -1)
0142:             rew = wp.to_torch(world.buffers["reward"]).to(args.learner_device, dtype=torch.float32).clone()
0143:             term = wp.to_torch(world.buffers["terminated"]).to(args.learner_device, dtype=torch.float32).clone()
0144:             trunc = wp.to_torch(world.buffers["truncated"]).to(args.learner_device, dtype=torch.float32).clone()
0145:             t_reason = wp.to_torch(world.buffers["termination_reason"]).to(args.learner_device).clone()
0146:             s_reason = wp.to_torch(world.buffers["safety_reason"]).to(args.learner_device).clone()
0147:             # Per-world overflow may be cleared by selective reset. Check EVERY
0148:             # simulated tick, not merely every tenth tick, before any auto-reset.
0149:             require_no_warp_overflow(world, i + 1)
0150:             result["overflow_checked_steps"] = i + 1
0151:             replay.add_batch(old, state, action, rew, nextobs, nextstate, term)


## 02 Warp shader compilation
Evidence: 05 original shield launch timed out 85s; 06 compiled in 2.57s
Source excerpt:
research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v80_shield.py:24-40
0024:
0025: if HAS_WARP:
0026:     # Non-differentiable environment transition, not a policy gradient layer.
0027:     # Avoid unused backward codegen and large 4x16 nested loop unrolling.
0028:     wp.set_module_options({"enable_backward": False, "max_unroll": 2})
0029:
0030:     @wp.func
0031:     def _clamp_vel_v80(
0032:         vx_prev: float, vy_prev: float, vx_cmd: float, vy_cmd: float,
0033:         dt_s: float, v_max: float, a_max: float,
0034:     ) -> wp.vec2:
0035:         dvx = vx_cmd - vx_prev
0036:         dvy = vy_cmd - vy_prev
0037:         dv_norm = wp.sqrt(dvx * dvx + dvy * dvy)
0038:         max_dv = a_max * dt_s
0039:         if dv_norm > max_dv and dv_norm > 1.0e-12:
0040:             scale = max_dv / dv_norm


## 03 Sensor kernel signature
Evidence: 06 launch passed 13 args but kernel needed 14; 07 progressed
Source excerpt:
research/zhang_xue/src/zhang_xue/gpu/mission_v80.py:62-78
0062:         ])
0063:         self._update_phy()
0064:         self._launch(_sense_visible_v79, (self.envs, 6), [
0065:             n.state, w["positions"], w["truth"], w["episode_step"], w["episode_id"],
0066:             w["battery"], w["unsafe_mask"], w["failed"], self.build_wp, len(self.buildings),
0067:             int(p.seed), float(p.pd), float(p.pf), float(p.ds_aoi_eta),
0068:         ])
0069:         self._launch(_reward_v79, self.envs, [
0070:             n.state, w["truth"], w["team_u_prev"], w["gcs_tp"], w["gcs_fp"], w["gcs_fp_ever"],
0071:             w["battery"], w["episode_step"], w["failed"], w["intervention"], w["reward"],
0072:             w["terminated"], w["truncated"], w["done"], w["termination_reason"],
0073:             self.logmf, self.logme, float(p.ds_aoi_eta), float(p.confirm_th),
0074:             float(p.ds_uncertainty_ceiling), float(p.r_tp), float(p.r_fp), float(p.fp_hold),
0075:             int(p.enable_fp_unlatch), float(p.safety_failure_penalty), float(p.w_aoi),
0076:             float(p.w_loc_team), float(p.w_blocked_motion), float(p.lambda_b),
0077:             float(p.continuous_aoi_penalty_w), float(p.t_ref_s), int(p.horizon),
0078:         ])


## 04 Termination buffer key
Evidence: 07 KeyError terminal_reason; 08 world.step PASS
Source excerpt:
research/zhang_xue/src/zhang_xue/gpu/mission_v80.py:68-80
0068:         ])
0069:         self._launch(_reward_v79, self.envs, [
0070:             n.state, w["truth"], w["team_u_prev"], w["gcs_tp"], w["gcs_fp"], w["gcs_fp_ever"],
0071:             w["battery"], w["episode_step"], w["failed"], w["intervention"], w["reward"],
0072:             w["terminated"], w["truncated"], w["done"], w["termination_reason"],
0073:             self.logmf, self.logme, float(p.ds_aoi_eta), float(p.confirm_th),
0074:             float(p.ds_uncertainty_ceiling), float(p.r_tp), float(p.r_fp), float(p.fp_hold),
0075:             int(p.enable_fp_unlatch), float(p.safety_failure_penalty), float(p.w_aoi),
0076:             float(p.w_loc_team), float(p.w_blocked_motion), float(p.lambda_b),
0077:             float(p.continuous_aoi_penalty_w), float(p.t_ref_s), int(p.horizon),
0078:         ])
0079:         self._observe()
0080:         return w["obs"], w["mask"], w["reward"], w["terminated"], w["truncated"], w["done"]


## 05 Order of physical clamps
Evidence: 15 max CPU-GPU vel diff 8.228706m/s vs16 1.4166e-6m/s
Source excerpt:
research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v80_shield.py:58-74
0058:     ):
0059:         e, u = wp.tid()
0060:         cx = wp.clamp(act[e, u, 0], -1.0, 1.0) * v_max
0061:         cy = wp.clamp(act[e, u, 1], -1.0, 1.0) * v_max
0062:         # Match CPU clamp_velocity_with_accel: cap acceleration FIRST, then speed.
0063:         # Pre-normalizing an over-speed command here changes direction/acceleration.
0064:         v_nom = _clamp_vel_v80(vel[e, u, 0], vel[e, u, 1], cx, cy, dt_s, v_max, a_max)
0065:         cand[e, u, 0] = v_nom[0]
0066:         cand[e, u, 1] = v_nom[1]
0067:
0068:     @wp.func
0069:     def _geom_physical_flags_v80(
0070:         x: float, y: float, vx: float, vy: float,
0071:         b: wp.array2d(dtype=wp.float32), n_b: int, dt_s: float,
0072:     ) -> int:
0073:         """Exact swept physical geometry detector (1=boundary, 2=building)."""
0074:         nx = x + vx * dt_s


## 06 dt_s time mismatch
Evidence: 17 TDD RED then 18 GREEN
Source excerpt:
research/zhang_xue/src/zhang_xue/gpu/mission_v80.py:16-32
0016: class GPUWorldV80Params(GPUWorldV79Params):
0017:     dt_s: float = 1.0
0018:     strict_braking_terminate: bool = False  # False = CPU trajectory_safety_violations parity
0019:
0020:     def validate(self) -> None:
0021:         super().validate()
0022:         # V79 fixed MAC duration, sensing sampling and physical shield assume dt=1s.
0023:         # Reject alternative dt instead of silently simulating a different experiment.
0024:         if not math.isfinite(self.dt_s) or abs(self.dt_s - 1.0) > 1.0e-12:
0025:             raise ValueError("V80 Warp runtime requires dt_s=1.0; MAC and safety kernels use a 1s tick")
0026:
0027:
0028: class WarpParallelMissionV80(WarpParallelMissionV79):
0029:     """V80 Warp GPU runtime with feasible lateral evasion + 3-sweep coordinate descent shield."""
0030:
0031:     def __init__(self, params: GPUWorldV80Params):
0032:         super().__init__(params)


## 07 Overflow guard placement
Evidence: 19 failed harness; 20 CPU injection PASS; 37 final 3 steps guarded
Source excerpt:
scripts/gpu/run_v80_real_rollout.py:23-39
0023: ROOT = Path(__file__).resolve().parents[2]
0024: sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]
0025:
0026:
0027: def require_no_warp_overflow(world, vector_step: int) -> None:
0028:     """Fail closed on any MAC event overflow before the earliest auto reset."""
0029:     import warp as wp
0030:     count = int(wp.to_torch(world.network.data["overflow"]).sum().item())
0031:     if count:
0032:         raise RuntimeError(
0033:             f"Warp packet event/gen overflow at vector_step={vector_step}: "
0034:             f"{count} events before selective auto-reset"
0035:         )
0036:
0037:
0038: def main() -> int:
0039:     parser = argparse.ArgumentParser(description=__doc__)


## 08 Unsafe state committed after failure
Evidence: 29 TDD RED x=8 to x=17; 32 CPU PASS and 34 CUDA PASS
Source excerpt:
research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v80_shield.py:258-274
0258:                         unsafe_mask[e, peer] = wp.uint8(1)
0259:         safety_reason[e] = wp.uint8(s_flags)
0260:         if phys_fail != 0:
0261:             failed[e] = wp.uint8(1)
0262:             # Fail-closed transaction: an unsafe candidate MUST NOT be written
0263:             # to the physics state. Terminate through _reward_v79 and let the
0264:             # caller reset the world. Preserve pre-step velocities as well
0265:             # (instant zeroing would itself break the acceleration contract).
0266:             intervention[e] = 0.0
0267:             return
0268:         # Phase 3: commit positions, velocities, continuous intervention norm, and Zeng 2019 energy
0269:         interventions = float(0.0)
0270:         for u in range(N_UAV):
0271:             if battery[e, u] <= 0.0:
0272:                 vel[e, u, 0] = 0.0
0273:                 vel[e, u, 1] = 0.0
0274:                 continue


## 09 No formal feasible shield guarantee
Evidence: 38 final MATD3 38 safety; 39 MASAC 3 safety
Source excerpt:
research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v80_shield.py:159-175
0159:                 best_vx = v_brk[0]
0160:                 best_vy = v_brk[1]
0161:         # Candidates 2..65: 4 radii x 16 angles inside kinematic acceleration disk
0162:         max_dv = a_max * dt_s
0163:         for ri in range(4):
0164:             r = max_dv * (1.0 - 0.25 * float(ri))
0165:             for ai in range(16):
0166:                 ang = float(ai) * (6.283185307179586 / 16.0)
0167:                 vc = _clamp_vel_v80(vpx, vpy, vpx + r * wp.cos(ang), vpy + r * wp.sin(ang), dt_s, v_max, a_max)
0168:                 if _cand_feasible_v80(e, u, vc[0], vc[1], check_peers, pos, v_safe, battery, b, n_b, safe_d, a_max) != 0:
0169:                     dx = vc[0] - vnx
0170:                     dy = vc[1] - vny
0171:                     cost = dx * dx + dy * dy
0172:                     if cost < best_cost:
0173:                         best_cost = cost
0174:                         best_vx = vc[0]
0175:                         best_vy = vc[1]


## 10 Tight initial UAV spawn
Evidence: Spawn sqrt(65^2+30^2)=71.59m with 50m minimum
Source excerpt:
research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v79_reset_obs.py:159-175
0159:             gcs_tp[e, c] = wp.uint8(0)
0160:             gcs_fp[e, c] = wp.uint8(0)
0161:             gcs_fp_ever[e, c] = wp.uint8(0)
0162:         for u in range(N_UAV):
0163:             pos[e, u, 0] = 90.0 + 65.0 * float(u)
0164:             pos[e, u, 1] = 110.0 + 30.0 * float(u % 2)
0165:             pos[e, u, 2] = uav_alt
0166:             pos_xy[e, u, 0] = pos[e, u, 0]
0167:             pos_xy[e, u, 1] = pos[e, u, 1]
0168:             vel[e, u, 0] = 0.0
0169:             vel[e, u, 1] = 0.0
0170:             energy[e, u] = battery_max
0171:             unsafe_mask[e, u] = wp.uint8(0)
0172:         pos[e, N_UAV, 0] = 50.0
0173:         pos[e, N_UAV, 1] = 50.0
0174:         pos[e, N_UAV, 2] = 15.0
0175:         rng = wp.rand_init(seed + e * 4093 + ep_id[e] * 11939)


## 11 Actor lacks explicit five peers and NFZ map
Evidence: Observe channels 0..13 and pooled belief only
Source excerpt:
research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v79_reset_obs.py:197-213
0197:             net.tx_airtime[e, r] = 0
0198:         done[e] = wp.uint8(0)
0199:
0200:     @wp.kernel
0201:     def _observe_core_v79(
0202:         net: NetworkBuffersV67, dims: SimDimsV67, pos: wp.array3d(dtype=wp.float32),
0203:         vel: wp.array3d(dtype=wp.float32), batt: wp.array2d(dtype=wp.float32),
0204:         blackout_xy: wp.array2d(dtype=wp.int32), n_bo: int, blackout_r: int,
0205:         obs: wp.array3d(dtype=wp.float32), mask: wp.array3d(dtype=wp.uint8),
0206:         battery_max: float, qcap: float, v_max: float,
0207:     ):
0208:         e, u = wp.tid()
0209:         x = pos[e, u, 0]
0210:         y = pos[e, u, 1]
0211:         gx = pos[e, N_UAV, 0]
0212:         gy = pos[e, N_UAV, 1]
0213:         dx = (gx - x) / 1000.0


## 12 Fixed DS masses and eta aging
Evidence: 25 eta0 YAML rejected UQ8.8 saturation
Source excerpt:
research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml:21-37
0021: ds_me: 0.80
0022: confirm_th: 0.90
0023: ds_uncertainty_ceiling: 0.20
0024: ds_fusion_mode: sum
0025: ds_aoi_eta: 0.02
0026: aoi_ewma_end_to_end: true
0027: evidence_calibration: paper_fixed
0028: tx_power_w: 0.1
0029: mac_cca_threshold_dbm: -82.0
0030: mac_noise_power_w: 4.38957709293155e-13
0031: enable_tx_power_control: false
0032: enable_3d_sensing_occlusion: true
0033: enable_accel_energy: false
0034: enable_fp_unlatch: true
0035: obs_pool_mode: semantic
0036: peer_reward_mode: info_gain
0037: aoi_reward_mode: exponential
