# Source excerpts — part 2

## 13 Count-power and mass fusion differ
Evidence: Source _fuse_uq88_v79 297; no posterior-wide parity
Source excerpt:
research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v79_kernels.py:293-309
0293:                 net.ledger_neg[e, u, u, cid] = wp.uint16(wp.clamp(int(wp.round(n * 256.0)), 0, 65535))
0294:                 net.ledger_gen[e, u, u, cid] = wp.int16(t)
0295:
0296:     @wp.func
0297:     def _fuse_uq88_v79(
0298:         net: NetworkBuffersV67, e: int, receiver: int, cid: int, t: int,
0299:         logmf: float, logme: float, eta: float,
0300:     ) -> wp.vec3:
0301:         p = float(0.0)
0302:         n = float(0.0)
0303:         for org in range(N_UAV):
0304:             if net.ledger_gen[e, receiver, org, cid] < 0:
0305:                 continue
0306:             age = wp.max(0.0, float(t - int(net.ledger_gen[e, receiver, org, cid])))
0307:             weight = wp.exp(-eta * age) * (1.0 / 256.0)
0308:             p += float(net.ledger_pos[e, receiver, org, cid]) * weight
0309:             n += float(net.ledger_neg[e, receiver, org, cid]) * weight


## 14 Local team reward vs GCS only
Evidence: Reward source 337-357 local ledger and GCS separately
Source excerpt:
research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v79_kernels.py:344-360
0344:                 n += float(net.ledger_neg[e, u, u, cid]) * discount
0345:             if p + n > 0.0:
0346:                 ds = _ds_v79(p, n, logmf, logme)
0347:                 if ds[2] < prev_u[e, cid]:
0348:                     loc += w_loc * (prev_u[e, cid] - ds[2])
0349:                     prev_u[e, cid] = ds[2]
0350:             m = _fuse_uq88_v79(net, e, N_UAV, cid, t, logmf, logme, eta)
0351:             if m[0] >= confirm and m[2] <= uncert_ceiling:
0352:                 if truth[e, cid] != wp.uint8(0) and tp[e, cid] == wp.uint8(0):
0353:                     tp[e, cid] = wp.uint8(1)
0354:                     rx_t = float(net.gcs_last_rx_us[e, cid] - net.mission_epoch_us[e]) / 1.0e6
0355:                     gen_t = float(net.gcs_latest[e, cid])
0356:                     delay = wp.max(0.0, rx_t - gen_t)
0357:                     delivered += r_tp + w_aoi * wp.exp(-delay / ref_age)
0358:                 elif truth[e, cid] == wp.uint8(0) and fp[e, cid] == wp.uint8(0):
0359:                     fp[e, cid] = wp.uint8(1)
0360:                     if fp_ever[e, cid] == wp.uint8(0):


## 15 No explicit traffic penalty
Evidence: YAML lambda_b0 and w_p2p0; airtime penalty YAML supported
Source excerpt:
research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml:36-46
0036: peer_reward_mode: info_gain
0037: aoi_reward_mode: exponential
0038: e_max_j: 180000.0
0039: w_p2p: 0.0
0040: lambda_b: 0.0
0041:
0042: battery_termination_mode: all_depleted
0043:
0044: # V77 parameter sweep (illustrative, NOT calibrated RF material)
0045: rf_obstacle_model: multi_obstacle
0046: rf_per_building_loss_db: 12.0


## 16 PHY noise vs UavNetSim v1
Evidence: YAML4.389577e-13W vs paper4e-11W =19.596 dB
Source excerpt:
research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml:26-42
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
0038: e_max_j: 180000.0
0039: w_p2p: 0.0
0040: lambda_b: 0.0
0041:
0042: battery_termination_mode: all_depleted


## 17 Ray cylinder NFZ loss 12 dB heuristic
Evidence: Earlier V79 sweep 0 6 12 20 produces same differences
Source excerpt:
research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml:42-46
0042: battery_termination_mode: all_depleted
0043:
0044: # V77 parameter sweep (illustrative, NOT calibrated RF material)
0045: rf_obstacle_model: multi_obstacle
0046: rf_per_building_loss_db: 12.0


## 18 Warp network differs from upstream UavNetSim SimPy
Evidence: No per-packet end-to-end full parity logs
Source excerpt:
research/zhang_xue/src/zhang_xue/gpu/runtime_v67.py:117-133
0117:         raise ValueError("building radius/height must be positive; blackout centers integers")
0118:     return p, b, bo.astype(np.int32)
0119:
0120:
0121: class WarpCompressedRadioNetworkV67:
0122:     """Experimental EWMA packet radio; CUDA parity and RL integration unverified.
0123:
0124:     Generation-only ACK watermarks enforce strictly increasing per-cell sensing
0125:     generations. Raw-count mode is unsupported by this compressed layout.
0126:     """
0127:
0128:     @staticmethod
0129:     def active_layout_id() -> str:
0130:         return "v67_datamule_compressed"
0131:
0132:     @staticmethod
0133:     def active_soa_bytes_per_environment(


## 19 MATD3 and MASAC critic loss scales
Evidence: MASAC sum over independent critics; MATD3 distinct Q objective
Source excerpt:
uav_network/algorithms/masac/learner.py:300-316
0300:         if not self.share_critic_parameters:
0301:             # Hybrid-MASAC defines one batch-mean critic objective J_Q(theta_i)
0302:             # per agent. A single optimizer over independent modules must sum
0303:             # those objectives rather than average once more over agents.
0304:             critic_loss = critic_loss * len(self.critics)
0305:         critic_step = (lambda *args: self.precision.step("critic", *args, grad_clip_mode=a.grad_clip_mode)) if self.precision.enabled else (lambda *args: step_optimizer(*args, grad_clip_mode=a.grad_clip_mode))
0306:         critic_grad = critic_step(critic_loss, self.critic_optimizer, self.critics.parameters(), a.grad_clip_norm, collect_metrics)
0307:         self.updates += 1
0308:         # Passive loss collector; original MASAC critic step is unchanged.
0309:         dense_sink = getattr(self, "_dense_loss_sink", None)
0310:         if dense_sink is not None:
0311:             dense_sink.record("critic_loss", critic_loss)
0312:             dense_sink.record("critic_grad_norm", critic_grad)
0313:             dense_sink.record("td_error_abs_mean", (q1.detach()-target.detach()).abs().mean())
0314:         if not collect_metrics:
0315:             return {}
0316:         return scalar_metrics({**critic_metrics(q1, q2, target, data),


## 20 No RL convergence or mission completion
Evidence: 37 E4096 3 steps 2 updates; 38 and39 no 15-target success
Source excerpt:
scripts/gpu/run_v80_real_rollout.py:40-56
0040:     parser.add_argument("--algorithm", choices=["matd3", "masac"], required=True)
0041:     parser.add_argument("--scenario", type=Path, default=ROOT / "research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml")
0042:     parser.add_argument("--scenario-set", action="append", default=[])
0043:     parser.add_argument("--envs", type=int, default=64)
0044:     parser.add_argument("--steps", type=int, default=300)
0045:     parser.add_argument("--horizon", type=int, default=None)
0046:     parser.add_argument("--strict-braking-terminate", action="store_true")
0047:     parser.add_argument("--updates-per-step", type=int, default=1)
0048:     parser.add_argument("--batch-size", type=int, default=128)
0049:     parser.add_argument("--hidden", type=int, default=256)
0050:     parser.add_argument("--world-device", default="cuda:0")
0051:     parser.add_argument("--learner-device", default="cuda:0")
0052:     parser.add_argument("--seed", type=int, default=44)
0053:     parser.add_argument("--allow-experimental-physics", action="store_true")
0054:     parser.add_argument("--output", type=Path, required=True)
0055:     args = parser.parse_args()
0056:     args.output.parent.mkdir(parents=True, exist_ok=True)


## 21 GPU0 occupied and CUDA peer unavailable
Evidence: nvidia-smi GPU0 11.7 GiB consumed; all tests cuda1
Source excerpt:
scripts/gpu/run_v80_real_rollout.py:46-62
0046:     parser.add_argument("--strict-braking-terminate", action="store_true")
0047:     parser.add_argument("--updates-per-step", type=int, default=1)
0048:     parser.add_argument("--batch-size", type=int, default=128)
0049:     parser.add_argument("--hidden", type=int, default=256)
0050:     parser.add_argument("--world-device", default="cuda:0")
0051:     parser.add_argument("--learner-device", default="cuda:0")
0052:     parser.add_argument("--seed", type=int, default=44)
0053:     parser.add_argument("--allow-experimental-physics", action="store_true")
0054:     parser.add_argument("--output", type=Path, required=True)
0055:     args = parser.parse_args()
0056:     args.output.parent.mkdir(parents=True, exist_ok=True)
0057:     result = {
0058:         "status": "RUNNING", "version": "V80", "envs": args.envs, "steps": args.steps,
0059:         "algorithm": args.algorithm, "batch_size": args.batch_size,
0060:         "world_device": args.world_device, "learner_device": args.learner_device,
0061:         "feasible_66cand_shield": True, "strict_braking_terminate": bool(args.strict_braking_terminate),
0062:         "updates": 0, "world_steps": 0, "episodes_ended_total": 0,


## 22 900-step reset demonstrated zero action only
Evidence: 24 gate 4x903 static 4 truncations at step900
Source excerpt:
scripts/gpu/run_v80_real_rollout.py:148-164
0148:             # simulated tick, not merely every tenth tick, before any auto-reset.
0149:             require_no_warp_overflow(world, i + 1)
0150:             result["overflow_checked_steps"] = i + 1
0151:             replay.add_batch(old, state, action, rew, nextobs, nextstate, term)
0152:             world.auto_reset_gpu()
0153:
0154:             ep_done = int((term + trunc).clamp(max=1.0).sum().item())
0155:             result["episodes_ended_total"] += ep_done
0156:             step_tr = {
0157:                 "safety_failed": int(((t_reason & 1) != 0).sum().item()),
0158:                 "all_uavs_depleted": int(((t_reason & 2) != 0).sum().item()),
0159:                 "all_targets_confirmed_at_gcs": int(((t_reason & 4) != 0).sum().item()),
0160:                 "horizon_truncated": int(((t_reason & 8) != 0).sum().item()),
0161:             }
0162:             step_sr = {
0163:                 "boundary_collision": int(((s_reason & 1) != 0).sum().item()),
0164:                 "building_collision": int(((s_reason & 2) != 0).sum().item()),


## 23 eta0 and highnoise YAML unsupported
Evidence: 25 audit reports eta0 overflow and highnoise any_depleted
Source excerpt:
research/zhang_xue/configs/scenario_v79_static_map_eta0_1km.yaml:21-37
0021: ds_me: 0.80
0022: confirm_th: 0.90
0023: ds_uncertainty_ceiling: 0.20
0024: ds_fusion_mode: sum
0025: ds_aoi_eta: 0.0
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


## 24 Every step host D2H sync and JSONL writing
Evidence: Runner writes progress and synchronous counters per step
Source excerpt:
scripts/gpu/run_v80_real_rollout.py:183-199
0183:                     step_result["updates"].append(metrics)
0184:                     result["updates"] += 1
0185:             statuses.append(step_result)
0186:             result["world_steps"] = i + 1
0187:             args.output.with_suffix(".progress.jsonl").open("a").write(json.dumps(step_result, allow_nan=False) + "\n")
0188:
0189:         result["final_world_invariants"] = world.guard()
0190:         result["cuda_peak_memory_bytes"] = {
0191:             dev: {
0192:                 "max_allocated": int(torch.cuda.max_memory_allocated(0 if dev == "cuda" else int(dev.split(":")[1]))),
0193:                 "max_reserved": int(torch.cuda.max_memory_reserved(0 if dev == "cuda" else int(dev.split(":")[1]))),
0194:             }
0195:             for dev in sorted(set((args.world_device, args.learner_device)))
0196:             if dev == "cuda" or dev.startswith("cuda:")
0197:         }
0198:         result["status"] = "V80_FEASIBLE_SHIELD_GPU_ROLLOUT_PASS"
0199:         result["final_record"] = statuses[-1]
