# V124 — DS-only, tất cả conflict, bản vá có giới hạn và kiểm thử (28/11/2026)

## Điều chỉnh yêu cầu
Không thêm action. Canonical `act_dim=2` liên tục (chỉ vận tốc XY), độ cao z=50m, TX power 0.1W cố định, routing pure GCS deterministic. Không sửa 30 file `uav_network/algorithms/` hoặc 2 active YAML. Mọi thay đổi là opt-in/kiểm chứng, không huấn luyện baseline MATD3/MASAC.

## Sửa nguồn V124 (mỗi sửa có root cause)
1. C85/C96: sensor DS ngầm giả định repeated samples independent. CPU sensor loop và Warp `_sense_visible_v79` có `sensor_revisit_cooldown_steps` với mặc định `0` (back-compatible); >0 tránh ghi nhận cùng UAV/ô trước khi hết khoảng tick tối thiểu; giảm sample correlation chứ không chứng minh independence.
2. C95: chống duplicate cùng U/C/timestamp, invalid id, time rewind trong research-only DScorrelationMitigator. Không tuyên bố helper thay thế state production trên GPU.
3. C94/C97: pilot camera trước đó có sequential RNG confounding do số ô nhìn thấy khác. Chế độ CPU `sensor_rng_mode=counter_keyed` tạo CRN theo (seed,episode,tick,UAV,cell), FAIL-CLOSED nếu cấu hình này lên Warp; kịch bản chuẩn vẫn sequential RNG.
4. C86/C99: `v124_freeze_readiness_gate.py` chỉ in BLOCKED_NOT_FROZEN, kể cả khi gate hình thức được điền; cần độc lập nghiệm thu. 8 blocker G01–G08 trong JSON release.
5. C90/C93/C98: bổ sung chứng cứ test GPU mới, không thay thuật toán: V124-port kernel cooldown CPU+CUDA [22,22,22,44] khi 3-step cooldown; 4096x900 moving gentle_patrol no-reset PASS; RFstress v105 peer-enabled 4096x900 RESET-MODE PASS. Đây là bounded gate, KHÔNG chứng minh CPU/GPU full-world parity hoặc UavNetSim SimPy parity.

## Không được đánh tráo phạm vi test
- Bộ test Vast `644 passed / 6 skipped / 153 subtests` và GPU tests chạy trên nhánh **V122+port mã cooldown kernel V124**, KHÔNG phải full clean ZIP V124 toàn bộ source. Có một lượt FAIL 2 manifest do chưa cập nhật SHA, sau `--write` PASS.
- CPU full suite trên ChatGPT container không chạy hoàn tất: thiếu `warp`, timeout, một số failures; giữ nguyên log, KHÔNG ghi tất cả PASS.
- CPU V124 targeted 36/36 pass. Pilot camera CRN 5-seed×60-step: nadir GCS TP4.0, east1.0, sweep1.2; không thêm action.
- GPU v105 stress mode reset-enabled (terminal events4096); khác test no-reset moving Base. Không có benchmark actor learned.

## Quyết định kết thúc vòng này
`SCENARIO_FROZEN=false`; `PHASE2_BASELINE_STARTED=false`. Bảo toàn source vanilla MATD3/MASAC để Phase2 sau nghiệm thu. 8 blocker chưa khép là temporal DS dependence, sensor Pd/Pf, RF calibrated, UavNetSim parity, whole-world CPU/GPU parity, non-oracle missions, terminal success semantics, algorithm contract.

## Tái lập
```
export PYTHONPATH=research/zhang_xue/src:.
python -m pytest -q tests/test_v124_ds_duplicate_provenance.py tests/test_v124_ds_revisit_and_crn.py tests/test_v124_freeze_readiness_gate.py tests/test_v123_gimbal_action_contracts.py
python scripts/maintenance/verify_source_manifests.py --check
python scripts/maintenance/check_project_policy.py --scope full
python scripts/maintenance/v124_freeze_readiness_gate.py --output outputs/v124_conflict_audit/V124_FREEZE_RECHECK.json  # exit 2 expected
```
RTX3060 GPU port exact reproduction scripts and logs are documented in `V124_CPU_AND_GPU_LOG_MANIFEST.json`. Source V123 previous ZIP retained under `input/V123_ORIGINAL_UNMODIFIED.zip` (it itself contains original V122 ZIP). Generated code diff and SHA manifests included.

## Science references (primary)
- Zhang et al. IEEE T-IV 2024 https://doi.org/10.1109/TIV.2024.3352581
- Denoeux 2008 dependent evidence https://www.sciencedirect.com/science/article/pii/S0004370207001063
- UavNetSim source https://github.com/Zihao-Felix-Zhou/UavNetSim ; Zhou et al. https://arxiv.org/abs/2507.09852
- Zeng, Xu, Zhang 2019 energy https://doi.org/10.1109/TWC.2019.2902559
- Zwick et al. 2023 sensor model https://doi.org/10.3390/s23020664
