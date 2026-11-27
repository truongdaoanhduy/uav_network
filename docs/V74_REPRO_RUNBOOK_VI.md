# V74 reproducibility: nguồn, bản sửa và chạy lại

## Trạng thái và bảo vệ dữ liệu

- Gốc v73: `/home/aduy/Downloads/file.ipynb`, notebook chứa một ZIP source nhúng 445,953 bytes, SHA256 `1d474578dcc465b9aa55781c5a10bedc0e0a55bab2699923c82cdd7490f0ced0` (chưa xác minh GitHub SHA được ghi trong notebook). Không thay notebook gốc.
- Làm việc trên bản **clone-archive độc lập** `/home/aduy/Documents/NCKH/uav_network_v73_resume_20261126`, branch local `audit/v73-continuation-20261126`, không đụng Git `main` tại `/home/aduy/Documents/NCKH/uav_network`, **chưa push**.
- `key.txt`: tại repo main hiện tên liên tục `account_1 ... account_5`; riêng active account_2 là nhãn người dùng đã chọn cho notebook v73. Không đưa token, key backup, Kaggle config, W&B/HF credential vào report/ZIP. Không claim đã thay token account3 do người dùng chưa cấp token mới.
- CPU functional v74 chạy bằng `/home/aduy/miniconda3/bin/python`; Kaggle T4x2 CUDA không được điều khiển ở bước v74 này.

## Thay đổi source code

1. `uav_network/experiments/radio_config.py` **mới**: nguồn duy nhất chuyển scenario vào radio/Warp, validate six compiled MAC timing fields, ghi comparability CPU64/Warp32.
2. `uav_network/experiments/capacity_probe.py`: `scenario_overrides`, valid YAML, `radio_config.json`, network args và PHY step args đúng, preserve requested/effective.
3. `scripts/gpu/run_capacity_probe.py`: thêm `--scenario-set`, tránh lẫn learner `--set`.
4. `scripts/gpu/packet_workload.py`: `step_kwargs` cho 3 event steps và no-peer/direct-GCS ablation.
5. `tests/test_v74_radio_config_contract.py`: 7 test RED→GREEN; malformed compiled MAC timing fail closed.
6. `SOURCE_SHA256_MANIFEST.json`, `research/zhang_xue/SHA256_MANIFEST.json`: regenerate sau khi source/docs được chốt.

## Lệnh trực tiếp và log đính kèm

```bash
export PYTHONPATH="$PWD:$PWD/research/zhang_xue/src"
python -m pytest -q tests/test_v74_radio_config_contract.py
python -m pytest -q tests
python scripts/maintenance/check_project_policy.py --scope full
python scripts/maintenance/verify_source_manifests.py --check

# Functional tests CPU dùng overrides thực
python scripts/gpu/run_capacity_probe.py --mode functional --algorithm matd3 --network-device cpu --learner-device cpu --functional-envs 2 --functional-batch 4 --functional-hidden 32 --scenario-set tx_power_w=0.15 --scenario-set mac_cca_threshold_dbm=-80 --scenario-set compact3_decentralized_peer_select=false --scenario-set enable_peer_dissemination=false --output-dir outputs/demo_matd3
python scripts/gpu/run_capacity_probe.py --mode functional --algorithm masac --network-device cpu --learner-device cpu --functional-envs 2 --functional-batch 4 --functional-hidden 32 --scenario-set tx_power_w=0.15 --scenario-set mac_cca_threshold_dbm=-80 --scenario-set compact3_decentralized_peer_select=false --scenario-set enable_peer_dissemination=false --output-dir outputs/demo_masac

# Kaggle GPU thật: chọn T4 x2 + Internet + Run All trong notebook v73 (đính kèm nguyên bản)
# Chưa có full GPU env train; không dùng --capacity CPU giả làm GPU.
```

`outputs/qa_evidence_v74/` trong ZIP gồm RED test log, GREEN test log, `pytest_full.log` thất bại do manifest cũ, `pytest_full_after_manifest.log` nếu chạy trước sửa cuối, và log kiểm tra cuối, các output Matd3/Masac (`result.json`,`radio_config.json`,`requested/effective_config.json`,`resource.jsonl`,`progress.json`) và CSV/metadata nếu có. Nếu không có `uav_v73_logs.zip` phải coi số Kaggle trước đó chỉ do người dùng tường thuật. Bất cứ kết quả CPU nào cũng không chứng minh CUDA G2+, GPU full rollout hoặc scientific convergence.

## Port Warp mới đúng hướng, không thay đổi nhiệm vụ

- Hai gói khác nhau: **radio packet engine** và **full vector UAV environment**. Không gọi first là second.
- `wp.from_torch/to_torch` có thể share tensor storage trong cùng device; phải giữ tensor life + CUDA stream, không tạo NumPy tensors từng step.
- GPU rollout: preallocate `(E,U,...)` arrays; base Python host chỉ schedule/batch, Warp kernel update từng timestep trên CUDA. Keep per-environment `done_mask`, per-env RNG and reproducibility.
- Sense evidence generation có timestamp và FoV; app delivery phải giữ đúng origin/generation, AoI và TTL; ảnh hưởng reward chỉ từ accepted packet at GCS.
- `terminated` là sự kiện mission/safety/battery; `truncated` là horizon; learner bootstrapping/training mask xử lý riêng.
- Không thay đổi MATD3 và MASAC loss, critic/mask semantics trong migration; test gradient norms và output saturation trên batch thật.
