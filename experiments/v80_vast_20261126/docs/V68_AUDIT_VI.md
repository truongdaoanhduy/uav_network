# v6.8 — Kiểm định source, bằng chứng test và đánh giá logic học thuật UAV

**Phạm vi:** Mở trực tiếp ZIP v6.7 của người dùng; kiểm tra code thực trong 173 file; đối chiếu UavNetSim source và các bài báo. Thay đổi bảo thủ với regression RED→GREEN. Đây **không** phải công bố thuật toán mới và không có kết quả policy hội tụ.

## Cách đọc bằng chứng test

Sau `bash setup.sh --qa --seed 44`, xem `outputs/qa_evidence/SUMMARY.md`, `results.json`, `pytest_junit.xml`, `logs/*.txt` và `red_green/*.txt`. `results.json` chứa chính xác argv, đường dẫn, exit code, phiên bản Python/Torch/Warp, môi trường CUDA và ngày UTC. Trong ZIP, `outputs/qa_evidence/` được đóng gói **có chủ đích để kiểm định**, nhưng `.gitignore` loại khỏi Git. Test RED là những bài test cố ý thất bại *trước bản vá*, không phải lỗi của bản cuối. GPU exit=3 được ghi `BLOCKED`, không ghi PASS.

## Ma trận học thuật: vấn đề, nguyên nhân, quyết định, bằng chứng

| Mã | Mức | Phân hệ | Phát hiện trong code v6.7 | Xử lý v6.8 / bước tiếp theo |
|---|---|---|---|---|
| B01 | P0 | **NVIDIA Warp** | `gpu/runtime_v65.py`: `WarpCompressedRadioNetwork` mới cấp phát array; **không có `step()` xử lý packet/ACK/TTL**. SoA bytes/env là schema, không phải simulator hoạt động. | **BLOCKED** ở `q1_gpu_preflight.py`; viết Warp packet-event kernel end-to-end, đối chiếu trace CPU→GPU mới được train. |
| B02 | P0 | **Warp data-mule VRAM** | Compressed SoA không chứa `ack_wm_origin_pos/neg/ts` [src,dst,origin,cell]. CPU dùng 2×uint16+int32 = **8 byte/phần tử**. Với 6 UAV/400 cell: 6×7×6×400×8=**806.400 B/env thiếu**. | Chưa sửa GPU; cần thiết kế lại state sparse/per-origin, kiểm thử no-double-count và đo `torch.cuda.max_memory_allocated`, `nvidia-smi`, Warp allocation thật. |
| B03 | P0 | **UavNetSim event parity** | CPU dùng queue64, Warp 32, UavNetSim gốc queue200; CPU đánh giá interference khác time-overlap-weighting của upstream; Sionna RT không được port. | Không tuyên bố "khớp hoàn toàn". Replay cùng packet seed, channel, CCA, backoff, ACK, retry, TTL; so traces, PDR, E2E-delay, collision/queue. |
| B04 | P1, **FIX** | **Scaling path-vector** | `communication/delivery.py` dùng `MAX_HOPS=6` ngay cả khi N=8..30; hợp lệ 7-hop bị drop. | Thay bằng `self.radio_n - 1`, giữ N=6 bitwise; test v6.8 bao gồm 8 UAV và 6 UAV. **Chỉ xác minh route logic, chưa xác minh 7-hop full packet path**. |
| B05 | P1, **FIX** | **AoI time units** | `dt_s` có thể là 0.5 nhưng `t_step_s=self.t+1`, generation timestamp và mission age vẫn là số bước như giây. | Config **chỉ chấp nhận 1.0s** tới khi thuật toán đổi toàn bộ clock/AoI sang seconds; test nhận diện sai đơn vị. |
| B06 | P1, **FIX** | **AoI UQ8.8** | `encode_uq88` clamp ở ~255.996 evidence counts, trong khi config cho phép `eta=0` hoặc quá nhỏ suốt 900 step. | Kiểm tra bound \(\sum_{j=0}^{H-1}e^{-\eta j}\le 65535/256\); báo lỗi sớm thay vì im lặng mất bằng chứng. |
| B07 | P1 | **Reward & mission** | `r_gcs_aoi` thưởng *một lần lúc xác nhận TP* ở `environment/transition.py`; metric mean GCS mapwide AoI là censored convention, không phản ánh refresh dài hạn. | Giữ baseline làm **freshness-at-confirmation**, không viết luận văn là continuous AoI minimization. Nếu muốn long-term AoI, cần đổi objective riêng + matched ablation. |
| B08 | P1 | **Belief DS / false positives** | `ds_mf=ds_me=0.80`, `confirm_th=0.90`, `pd=0.85`, `pf=0.03` là các giả định khác mô hình sensor mà paper khác có thể dùng. `ds_fusion_mode=sum`+AoI UQ8.8 không phải tái lập y hệt Zhang 2024. | Thử ROC/precision–recall và false confirmation với threshold 0.90/0.99, fixed/calibrated DS; báo rõ mới là proposed extension. |
| B09 | P1 | **Metrics** | Runner `run_scenario.py` CPU chỉ báo `gcs_tp_confirmed`, `delivered_cell_coverage_pct` (ô có dữ liệu ở GCS), năng lượng/AoI; **không có full per-episode discovery%, true area coverage, discovery AUC** cho kịch bản mới. | Cần định nghĩa riêng target discovery (local DS), area observed, GCS delivery, time-to-confirm, AoI/PDR; xuất per-episode + bootstrap CI trước paper. Không dùng delivered-cell coverage thay map coverage. |
| B10 | P1 | **Benchmark map 1/2/3km** | Hình học 2/3km tỷ lệ XY nhưng chiều cao tòa nhà, altitude và RF distance chưa tỷ lệ; không phải những benchmark "cùng độ khó". | So nhóm geometry-similar và mission-realistic riêng, kiểm soát mật độ target, FoV/altitude, seed, số UAV. Quét link SINR **không** phải PDR. |
| B11 | P1 | **No-send/peer trade-off** | `lambda_b=0`, `w_p2p=0` có thể khiến `NO_SEND` ít hữu dụng ở low traffic; `SHARE_PEERS` truyền multi-origin nhưng policy RL chưa chứng minh gain. | Giữ ON/OFF và learned-vs-always-send baselines; đo offered traffic, collision, PDR, queue, delivered AoI; bỏ action nếu ablation không có gain có ý nghĩa. |
| B12 | P1 | **RL convergence/validity** | Gate MATD3/MASAC chỉ **4 transitions + 1 optimizer update**, không có 4096 env GPU hay checkpoint hội tụ. `uav_network/experiments` có metric helper nhưng chưa nối đầy đủ vào scenario. | Cần train nhiều seed, eval 30–100 ep/seed, phương pháp kiểm định thống kê, baseline đối chứng, kiểm tra mask và action saturation. |
| B13 | P1 | **Energy/safety** | UAV động học 2D ở altitude cố định, mô hình pin Zeng steady-level, braking shield xấp xỉ; `battery_termination_mode=all_depleted` thay nghĩa mission so với `any_depleted`. | Đánh giá safety violation, interventions, endurance theo tốc độ; không trộn termination modes giữa algorithms. |
| B14 | P2 | **Communication snapshot AoI** | UQ8.8 fixed-point và gen timestamp theo giây có lượng tử hóa, chưa chứng minh tương đương toàn bộ phép đo Xue 2026/packet AoI trong trường hợp collision/out-of-order/TTL. | So reference offline per-measurement theo seed; event trace cho stale/duplicate/retries before claiming exact Eq.9–10. |
| B15 | P2 | **CPU↔Warp step clock** | `network_substeps` mặc định 1: MAC mô phỏng một giây khi vị trí không đổi, sau đó UAV mới di chuyển. Substeps=2 cập nhật midpoint nhưng chưa có GPU parity. | Ablation 1 vs 2 vs finer substeps ở tốc độ cao; đo ảnh hưởng route/PDR/safety; không tự nâng fidelity mà thiếu benchmark. |
| B16 | P2 | **Source/package hygiene** | ZIP v6.7 có source + reports, **không có full raw pytest/JUnit logs**. | Thêm `scripts/maintenance/run_qa_bundle.py`, `setup.sh --qa`, `outputs/qa_evidence/` vào ZIP này; SHA manifest giữ source-only. |

**Kiểm tra bộ nhớ quan trọng:** Với N=10, grid40×40, riêng watermark đầy đủ của data-mule cần 10×11×10×1600×8 = **14.080.000 byte/env**; tại 2.048 env ≈ **26,86 GiB**, chưa tính replay, model, kernels. Vì vậy không thể tuyên bố cấu hình 2.048 env trên T4 chỉ từ con số SoA nén v6.5. Đây là bound theo **dtype CPU hiện tại**, không phải peak VRAM đo được.

## Quyết định thiết kế có căn cứ

- **Giữ:** kịch bản 6 UAV/1GCS/15 static targets như profile chính, chính sách velocity 2D, fixed TX 0.1W, DS belief, multi-origin data-mule và metrics tìm kiếm/truyền tin phân biệt.
- **Giữ để ablate:** `NO_SEND`, `SHARE_PEERS`, battery termination, peer communication (ON/OFF), mức congestion và reward AoI; chỉ giữ thành contribution nếu kết quả nhiều seed khẳng định.
- **Không đưa vào đóng góp chính:** RF TX power learner, r_airtime gần 0, kỳ vọng zero-shot N-var mà chưa đo, CUDA 4096env chưa có kernel packet.
- **Giữ đúng tên:** Zhang 2024 = hướng belief-map reconnaissance/search; Xue 2026 = AoI+DS cooperative search; UavNetSim = simulator tham khảo, **không được nói code custom CPU/Warp tương đương nếu chưa đối chiếu event**.

## Cách chạy

```bash
bash setup.sh --check-only
bash setup.sh --qa --seed 44
cat outputs/qa_evidence/SUMMARY.md
python scripts/maintenance/verify_source_manifests.py --check
bash setup.sh --gpu-gate  # EXPECT exit=3; BLOCKED, not FAIL surprise
```

## Nguồn đối chiếu đã kiểm tra

1. [UavNetSim source `config.py`](https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/master/utils/config.py) — Ptx=0.1W, CCA=-82dBm, 802.11b noise −174dBm/Hz + NF7dB, queue200, max retries5.
2. [UavNetSim PHY channel](https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/master/phy/channel.py) — CCA trong `is_busy_for`, success nhận gói theo SINR tại `_deliver`, interference overlap weighting.
3. [Zhang et al. 2024, IEEE T-IV](https://doi.org/10.1109/TIV.2024.3352581) — DS belief map cho coverage và target localization, không phải paper packet-event UavNetSim.
4. [Xue et al. 2026, Electronics](https://doi.org/10.3390/electronics15091875) — AoI+DS search with constrained communication, không phải tiêu chuẩn bắt buộc UQ8.8 bit-accurate.
5. [Zhou et al. 2025, UavNetSim-v1](https://arxiv.org/abs/2507.09852) — SimPy event-driven FANET baseline; preprint/conference không tự được gắn nhãn journal Q1.
6. [NVIDIA Warp 1.18 runtime](https://nvidia.github.io/warp/v1.18/user_guide/runtime.html) — array allocation/graph capture không đồng nghĩa kernel xử lý packet đã được implement.
7. [SchedNet 2019](https://arxiv.org/abs/1902.01554) / [IC3Net 2019](https://arxiv.org/abs/1812.09755) — learned communication gating cần có giá trị message và congestion/budget, không mặc định có gain.

## Roadmap đúng thứ tự nghiên cứu

1. Chốt định nghĩa scientific metrics, matched baselines và mục tiêu (search→GCS delivery/AoI).
2. Đối chiếu CPU UavNetSim bằng event traces, sửa PHY/MAC/ACK/queue/time-overlap từng lỗi kèm regression.
3. Thiết kế lại GPU data-mule state có memory bound trên T4; thực hiện đúng `step()`, các kernels packet, chặn overflow.
4. Test CPU–Warp parity theo ngữ cảnh, seed; khởi động 32→256→1024→2048 env/GPU; đo peak VRAM, throughput, overhead.
5. Train MATD3/MASAC trên scenario khớp GPU, đánh giá nhiều seed, chạy ablation & CI, khóa config SHA256 trước paper.

**Không sử dụng key.txt/Kaggle hoặc tuyên bố đã chạy CUDA.** Việc thiếu GPU không phải lý do bỏ log; gate GPU **phải** hiện trạng thái BLOCKED trong bằng chứng.
