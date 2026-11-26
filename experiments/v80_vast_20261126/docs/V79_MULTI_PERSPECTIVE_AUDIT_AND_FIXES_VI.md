# V79 — KIỂM TOÁN ĐA GÓC NHÌN V78 (14 VẤN ĐỀ + 10 LỖI P0/P1 MỚI) & BẢN VÁ CODE V79

**Ngày:** 2026-11-26 (UTC+07). **Nguồn kiểm toán:** `UAV_FULL_CODE_KAGGLE_ALL_LOGS_RUN_EXPLANATION_EVIDENCE_20261126.zip` (817 tệp, V77 9/9 PASS + V78 15/15 PASS = 24/24 Kaggle jobs PASS).

## 1. Đánh giá 14 vấn đề (ISS-01..ISS-14) của V78: Đúng, Sai, hay Thiếu?

- **ISS-01..04, ISS-07, ISS-10..14 (ĐÚNG hướng)**: V78 đánh giá trung thực phạm vi smoke test (`4096 × 3 steps`, `episodes_ended=0`, `gpu_env_cpu_parity_proven=false`, `paper_ready=false`).
- **ISS-05 (ĐÚNG MỘT PHẦN — THIẾU nguyên nhân toán học)**: Critic loss tăng 2.6× (`59.75 → 156.98` MATD3, `211.94 → 532.74` MASAC) và MASAC cao hơn MATD3 `3.4×` KHÔNG phải ngẫu nhiên mà do 2 cơ chế trực tiếp trong code:
  1. **Lệch hệ số nhân loss `3.0×` giữa 2 learner** ở `grad_clip_mode="global"`: `matd3/learner.py:262-265` tính `2 × MSE`, còn `masac/learner.py:299-304` nhân `len(self.critics)=6` → `6 × MSE`.
  2. **Cold-start + ngoại lai MSE `+93.87`**: Bước 1 quét ~17 ô mới (`loc ≈ 6.85`), bước 2 quét trùng ô (`Δu = 0.16`, `loc ≈ 1.43`, gói 1 ở GCS mới đạt `m_t = 0.80 < 0.90`), bước 3 gói 2 về GCS xác nhận 1–3 mục tiêu (`+31`/mục tiêu → `target_abs_max = 93.87`), làm MSE nổ mạnh và tỷ lệ ReLU chết ở `q1_relu_0` tăng `49.8% → 63.5%` chỉ sau 1 update.
- **ISS-06 (ĐÚNG MỘT PHẦN — THIẾU)**: Ngoài việc tách hình học NFZ khỏi suy hao vật liệu, V78 bỏ sót xung đột `12.0 dB` (YAML) vs `20.0 dB` (`GPUWorldV78Params:353`). Đối chiếu ITU-R P.2109-2 / P.2040-3 và 3GPP TR 36.777: `multi_obstacle` là mô hình suy hao cộng dồn tất định (chưa có shadowing/nhiễu xạ), cần giữ làm sweep `0/6/12/20 dB`.
- **ISS-08 & ISS-09 (THIẾU lỗi PBRS & Oracle Reward Leakage)**: Với 15 mục tiêu tĩnh, suy giảm mũ $e^{-0.02 \Delta t}$ lên khối bằng chứng làm "quên" mục tiêu tĩnh; và `loc` cộng gộp trực tiếp `net.ledger_pos[e, u, u, cid]` của mọi UAV dù chưa truyền về GCS tạo **oracle credit leakage** trong reward.

## 2. 10 lỗi P0/P1 chí tử mới phát hiện trong `mission_v78.py` & `run_v78_real_rollout.py`

1. **[P0] Selective `reset(done)` làm trôi đồng hồ mạng `+0.6 s` trên 4095 world đang chạy** (`mission_v78.py:425`): `self.network.step(600000)` không nhận mask `done`!
2. **[P0] Ghi đè `pos` tại chỗ trong vòng kiểm tra va chạm cặp** (`mission_v78.py:145-165`): lệch hình học tới 15 m, bỏ qua UAV hết pin, thiếu braking clearance và thiếu kiểm tra cặp sau phanh.
3. **[P0] Lệch độ trễ GCS `+0.6 s`** (`mission_v78.py:263, 425`): không gán `mission_epoch_us = now` sau warmup 600 ms và không trừ `mission_epoch_us` khi tính `rx_t`.
4. **[P0] Lặp RNG cảm biến giữa các episode** (`mission_v78.py:189`): `wp.rand_init` thiếu `ep_id[e]`.
5. **[P0] Gộp `terminated` và `truncated` vào `done`** (`mission_v78.py:279`, `run_v78_real_rollout.py:108`): triệt tiêu bootstrap $Q(s_T, a_T)$ ở horizon 900 s (Pardo et al., ICML 2018).
6. **[P0] Thiếu `fp_hold` (`-0.10`), `enable_fp_unlatch` + `gcs_fp_ever`, và `safety_failure_penalty` (`-100.0`)** (`mission_v78.py:267-277`).
7. **[P0] `obs[5]` và `mask[1:3]` bỏ qua vùng blackout** (`mission_v78.py:297, 311, 316`); **`obs[11]` là nhị phân `{0,1}`** thay vì tỷ lệ liên tục `[0,1]`; **`obs[12] = 0.0`** (ISS-14).
8. **[P0] Lệch cấu hình `20.0 dB` vs YAML `12.0 dB`** (`mission_v78.py:353`).
9. **[P0] Che khuất cảm biến 2D `bh >= 50.0`** (`mission_v78.py:200`) thay vì giao tia 3D–trụ.
10. **[P1] Host-sync `done.numpy()` mỗi bước** + PA DC power hardcode `1.0 W` (`mission_v78.py:172`).

## 3. Bản vá V79 đã triển khai (giữ nguyên tương thích ngược V78)

- `research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v79_kernels.py`: Toàn bộ kernel sửa 10 lỗi trên (`step_v67_masked`, `_move_v79` 2 pha, `_sense_visible_v79` với `ep_id` & 3D ray-cylinder, `_reward_v79` tách `terminated`/`truncated` & đủ `fp_hold`/`safety_pen`, `_observe_core_v79` có blackout + `obs[11]` liên tục + `obs[12]` ISS-14).
- `research/zhang_xue/src/zhang_xue/gpu/mission_v79.py`: `GPUWorldV79Params.from_yaml()`, `WarpParallelMissionV79`, và `auto_reset_gpu()` (0 host-sync, masked 600 ms warmup).
- `scripts/gpu/run_v79_real_rollout.py`: Đọc trực tiếp từ YAML, chỉ truyền `terminated` vào replay buffer, hỗ trợ `--horizon 2` kiểm thử selective reset nhiều episode, ghi `cuda_peak_memory_bytes`.
- `tests/test_v79_audit_and_parity_contracts.py`: Bộ kiểm thử hợp đồng & hồi quy V79.

## 4. Lệnh chạy xác minh trên Kaggle 2× Tesla T4

```bash
export PYTHONPATH="$PWD:$PWD/research/zhang_xue/src"
# 1. Kiểm thử selective auto-reset & terminated/truncated qua nhiều episode (horizon=2)
python3 scripts/gpu/run_v79_real_rollout.py --algorithm matd3 --envs 64 --steps 10 --horizon 2 --batch-size 64 --world-device cuda:0 --learner-device cuda:1 --allow-experimental-physics --output outputs/v79/reset_gate_matd3.json
# 2. Chạy E=4096, B=8192 trên T4x2 với YAML đồng bộ 12 dB
python3 scripts/gpu/run_v79_real_rollout.py --algorithm matd3 --envs 4096 --steps 10 --batch-size 8192 --hidden 256 --world-device cuda:0 --learner-device cuda:1 --allow-experimental-physics --output outputs/v79/cuda4096_matd3.json
```
