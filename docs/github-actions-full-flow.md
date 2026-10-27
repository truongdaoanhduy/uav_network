# Full flow training → visualize bằng GitHub Actions

GitHub Actions giữ vai trò supervisor bên ngoài Kaggle. Máy local và sandbox chỉ cần dùng để launch root một lần; các bước tiếp theo chạy trên cloud.

## Trước khi launch

- Workflow [UAV External Supervisor](https://github.com/truongdaoanhduy/uav_network/actions/workflows/uav_external_supervisor.yml) trên `main` phải ở trạng thái `active`.
- `KAGGLE_ENABLED_ACCOUNTS` hiện là `account_01,account_02`. Chỉ dùng account có trong allowlist này cho full flow. `key.txt` local có thể có thêm account; không bỏ `--account` để tự chọn chúng.
- Đã đối chiếu identity giữa GitHub secrets và `key.txt`: `account_01` = `duytrngoanh`; `account_02` = `vuliu123456` (27/10/2026).
- Kiểm tra quota, accelerator, job đang chạy và credential datasets trước mỗi đợt training; không ghi đè/cancel job cũ.
- Source phải commit/push. `wandb/` được ignore để giữ log local và không chặn kiểm tra source sạch của launcher.

Kiểm tra quyền truy cập từ GitHub mà không tạo training:

```bash
gh workflow run uav_external_supervisor.yml --ref main -f dry_run=true
gh run list --workflow uav_external_supervisor.yml --limit 3
```

Đọc run vừa tạo: bước probe phải báo `ok` cho identity Kaggle, W&B và HF; bước reconcile hiển thị các lineage tìm được và không submit child ở chế độ dry-run. Đây là kiểm tra authentication/read access; không phải kiểm chứng quyền ghi hay một lần train → visualize thực tế.

## Ví dụ root launch: MASAC BASE, hybrid75, 20k

Lệnh này là ví dụ để dùng khi đã yêu cầu launch; tài liệu này không tự chạy nó:

```bash
python -m uav_network kaggle launch \
  --stage train --algorithm masac --account account_01 \
  --config configs/comparison-20k-hybrid75-1kb.yaml \
  --credential-dataset duytrngoanh/uav-cpu-wandb-credential \
  --gpus-per-run 2 --auto-continue
```

Với `account_02`, dataset canonical tương ứng là `vuliu123456/uav-cpu-wandb-credential`. Chọn `--algorithm matd3` cho MATD3, thêm `--set pql.enabled=true` cho PQL. Acceleration policy được launcher chọn theo thuật toán; giữ nguyên config khoa học. Mỗi root chỉ chạy một thuật toán.

## Khi nào tắt sandbox được

Sau launch, xác nhận cả ba điều:

1. Kaggle root thực sự `running`, W&B learner đã nhận progress.
2. Manifest root có `orchestration=github_actions`, `external_supervisor_required=true` và ref `gh-tra-g...` (validate dùng `gh-val-g...`).
3. Account đúng allowlist, identity đã khớp và workflow GitHub đang `active`.

Khi đó có thể tắt sandbox/laptop. Riêng trạng thái W&B `running` chưa xác nhận đủ ba điều trên. Các run cũ không có marker này không tự được nhận vào supervisor mới.

## Cloud sẽ làm gì tiếp

- GitHub cron kiểm tra mỗi 15 phút; GitHub có thể chạy trễ. Không cần giữ một Actions job mở suốt thời gian training.
- GPU segment dừng theo time budget, chưa đủ episode: tiếp tục từ full-state checkpoint; thiếu quota thì chờ quota hợp lệ trên cùng account.
- Đủ 20.000 episode, `episode_budget_complete=true`, `stop_reason=episode_budget`: submit Kaggle CPU evaluation 30 episode, seed 2000000044, tạo visualization và publish W&B/HF.
- Nếu crash: chỉ recovery khi có checkpoint resumable đã xác minh đúng lineage/commit/progress. Thiếu checkpoint thì báo blocked, không giả thành công hoặc warm-start rồi gọi đó là resume.
- CPU evaluation bị ngắt: dùng progress đã lưu để retry; ref xác định trước giúp tránh tạo job trùng.

Supervisor hoạt động độc lập với sandbox. Quota hoặc lỗi dịch vụ cloud vẫn có thể làm việc chuyển bước phải chờ; trạng thái và log xem trong Actions và W&B.

## Khi flow gặp lỗi

Mỗi lần reconcile ghi `supervisor-report.json`, có trong artifact `uav-supervisor-report` của Actions, lưu 14 ngày. Lỗi một account không chặn các account sau. `recovery_blocked`, `recovery_required`, `service_unavailable`, lỗi account hoặc reconcile làm invocation báo lỗi để dễ kiểm tra, không đổi trạng thái lịch sử của learner.

- `waiting_for_quota`: chờ, chưa coi là submit thành công và không đổi account.
- `checkpoint_missing` / `environment_state_missing` / `checkpoint_provenance_mismatch`: cần checkpoint có bằng chứng phù hợp; không dùng policy để giả exact resume.
- `recovery_service_unavailable`: lần kiểm tra sau thử lại khi dịch vụ hoạt động.
- `recovery_budget_exhausted`: ba phiên crash recovery liên tiếp đã dùng hết; giữ nguyên checkpoint và chờ kiểm tra. Budget chỉ reset sau một segment clean, có thể chặn cả handoff CPU sau lần crash thứ ba.
- `cancelled`: không tự khởi chạy lại.

Bước probe --check-services của manual dry-run hiện chỉ ghi console khi probe thất bại trước reconcile; trường hợp đó chưa có JSON riêng. Lỗi setup/runner cũng cần xem log Actions.

[Phân tích MASAC và ma trận xử lý các nhóm lỗi](superpowers/evidence/2026-10-27-general-failure-hardening.md).
