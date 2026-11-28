# Kiểm Toán Pháp Y V107 → Bản Vá Sạch V108 & Danh Sách Chốt `SCENARIO_FROZEN`

**Trạng thái hiện tại:** `NOT SCENARIO_FROZEN` (chờ 1 lượt xác nhận GPU cuối trên Vast.ai `cuda:0` cho đúng mã nguồn V108).  
**Toàn vẹn thuật toán Phase 2:** 30/30 tệp trong `uav_network/algorithms/` giữ nguyên byte-identical (không chỉnh sửa MATD3/MASAC).

---

## 1. Nguyên nhân & Lý do kỹ thuật các thay đổi V107 → V108

1. **Giữ và xác thực `exploration_reward_mode: first_visit` (từ V107):**
   - **Nguyên nhân:** Ở V106 (`ds_ignorance`), UAV đứng yên (`a=0`) vẫn tiếp tục nhận `r_loc_team > 0` ở bước 2 (`1.6533` CPU; `[1.76, 1.724445]` Warp) do đo lặp ép $m_U \to 0$ (hover-farming).
   - **Giải pháp:** Chế độ `first_visit` đặt `u_now = 0.0` (CPU `transition.py:328-330`) và `u_cov = 0.0` (Warp `mission_v79_kernels.py:403-405`) ngay sau lần quan sát cục bộ đầu tiên của ô, tách bạch hoàn toàn phần thưởng bao phủ không gian với niềm tin DS dùng để xác nhận mục tiêu tại GCS.
   - **Kiểm chứng S19 trên V108 (`outputs/v108/s19_s20_first_visit_cpu.json`, 120 bước, 3 seeds 44/45/46):**
     - `hover`: reward trung bình `42.0` (giảm mạnh so với `73.0` ở `ds_ignorance`, không còn rò rỉ hover), coverage `5.50%`, TP `1.0`.
     - `random`: reward `42.33`, coverage `5.67%`, TP `1.0`.
     - `explore`: reward `80.0` (`+90.5%` so với hover), coverage `9.00%`, TP `2.0`.

2. **Vá 4 lỗi vệ sinh phát hiện trong kiểm toán pháp y V107:**
   - `scripts/maintenance/v107_ds_first_passage.py`: bổ sung `out.parent.mkdir(parents=True, exist_ok=True)` để tránh `FileNotFoundError` khi chạy trên repo sạch.
   - `tests/test_v105_complete_scenario_audit.py`: bổ sung assertion đối xứng `p_stress.exploration_reward_mode == "first_visit"` cho kịch bản RF-Stress.
   - `research/zhang_xue/src/zhang_xue/environment/configuration.py`: cập nhật chú thích trường `exploration_reward_mode` liệt kê đầy đủ `'first_visit'`.
   - `README.md`: cập nhật đầu trang phản ánh đúng kịch bản active V108 (`obs_schema_v2: true` = **549** dims raw / **553** dims encoded cho learner canonical; `first_visit`; `tx_power_w = 0.1 W`).
   - Bổ sung `tests/test_v108_first_visit_verification.py` dùng `unittest` thuần (chạy được cả ở môi trường không cài `pytest`).

3. **Điểm quyết định còn mở (`confirm_th: 0.90` vs `0.99`):**
   - `v107_ds_first_passage.py` cho thấy dưới cảm biến Bernoulli độc lập ($P_d=0{,}85, P_f=0{,}03, m_f=m_e=0{,}80, u \le 0{,}20$), `confirm_th=0.90` và `0.95` đều có báo động giả tích lũy `9.565e-4` sau 900 bước (2 mẫu dương giả liên tiếp vượt ngưỡng), trong khi `0.99` giảm hơn **32×** (`2.958e-5`) và giữ true detect ở `97.45%` sau 10 bước.
   - V108 hiện **giữ nguyên `confirm_th: 0.90`** trong cả 2 YAML để không tự ý đổi hợp đồng kịch bản khi chưa có xác nhận của tác giả.

---

## 2. Bộ lệnh chạy cổng Vast.ai `cuda:0` cuối cùng để khóa `SCENARIO_FROZEN`

Bằng chứng trong `outputs/s24_vast/` thuộc V106 (SHA256 YAML cũ) và `full_pytest.log` từng lỗi `tomllib` ở bước collect. Chạy đúng 5 lệnh sau trên Vast.ai `cuda:0` với mã nguồn V108:

```bash
mkdir -p outputs/v108_vast

# 1. Governance & Manifests
python3 scripts/maintenance/verify_source_manifests.py --check | tee outputs/v108_vast/01_manifest.log
python3 scripts/maintenance/check_project_policy.py --scope full | tee outputs/v108_vast/02_policy.log

# 2. Full Pytest (toàn bộ 586+ tests sau bản vá tomli)
python3 -m pytest -q tests/ research/zhang_xue/tests/ | tee outputs/v108_vast/03_full_pytest.log

# 3. CUDA Compute Sanitizer trên kernel V108 (bao gồm test_v107_first_visit_runtime.py)
compute-sanitizer --tool memcheck python3 -m pytest -q tests/test_v107_first_visit_runtime.py tests/test_v102_cuda_reset_sanitizer.py | tee outputs/v108_vast/04_sanitizer.log

# 4. Endurance 4096 x 900 cho Base và RF-Stress trên YAML V108 (first_visit)
python3 scripts/gpu/v99_endurance_sticky_cuda_gate.py \
  --scenario research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml \
  --envs 4096 --steps 900 --device cuda:0 \
  --output outputs/v108_vast/05_endurance_base.json | tee outputs/v108_vast/05_endurance_base.log

python3 scripts/gpu/v99_endurance_sticky_cuda_gate.py \
  --scenario research/zhang_xue/configs/scenario_v105_ds_rf_stress_1km.yaml \
  --envs 4096 --steps 900 --device cuda:0 \
  --output outputs/v108_vast/06_endurance_rf_stress.json | tee outputs/v108_vast/06_endurance_rf_stress.log

# 5. Niêm phong SHA256SUMS toàn bộ bằng chứng
(cd outputs/v108_vast && sha256sum * > SHA256SUMS)
```

Khi cả 5 lệnh trên PASS (`0` errors, `PASS_STICKY_RESET_MODE`, `0` safety/overflow/nonfinite), kịch bản đủ điều kiện khóa **`SCENARIO_FROZEN`** và chuyển sang Phase 2 (**`S25` replay buffer VRAM + canonical `MATD3` & `MASAC`** từ `uav_network/algorithms/baselines/canonical_continuous.py`).
