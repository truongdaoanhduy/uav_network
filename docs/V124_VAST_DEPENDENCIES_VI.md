# V124 — Cài đặt dependency trên Vast.ai (28/11/2026)

## Kết luận kiểm toán

Bản phát hành GitHub V124 ban đầu **không có** `requirements.txt` ở gốc. Các dependency gốc trong `pyproject.toml` cho phép khoảng rộng: numpy>=1.24,<3; torch>=2.3,<3; PyYAML>=6,<7. Nhánh research chỉ khai báo NumPy/PyYAML, GPU extra khai báo Warp 1.18.0. Chỉ chạy `bash setup.sh --gpu` ở một máy mới có thể cài bản torch mới khác với bản đã thử thành công. V124 bổ sung `requirements.txt` và `requirements-plots.txt` để khôi phục bộ phiên bản thử nghiệm.

### Bộ cài đặt chuẩn trên Vast Linux x86_64

Yêu cầu: GPU NVIDIA hỗ trợ, driver NVIDIA **>=525**, Internet để tải các wheel; **Python 3.10** ưu tiên (Python 3.10.22 đã được xác minh). Lệnh kiểm tra đầu tiên:

```bash
nvidia-smi
python3 --version
uname -m
```

Bước cài đặt, chạy từ **gốc repository**:

```bash
git clone --branch audit/v124-ds-only-c01-c99-20261128 https://github.com/truongdaoanhduy/uav_network.git uav_network_v124
cd uav_network_v124
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
python -m pip install --no-deps -e . -e ./research/zhang_xue
python -m pip check
```

**Không chạy `bash setup.sh --gpu` sau quy trình này**, vì script cũ sẽ tạo lại venv và dùng các khai báo phiên bản rộng trong `pyproject.toml`. Khi cần vẽ đồ thị, chạy thêm `python -m pip install -r requirements-plots.txt`.

### Preflight bắt buộc

```bash
python - <<'PY'
import platform, numpy as np, torch, warp as wp, yaml, simpy, pytest
print("Python / host:", platform.python_version(), platform.machine())
print("NumPy:", np.__version__, "| PyYAML:", yaml.__version__)
print("PyTorch:", torch.__version__, "| built CUDA:", torch.version.cuda)
print("Warp:", wp.__version__)
print("SimPy:", simpy.__version__, "| pytest:", pytest.__version__)
assert torch.__version__ == "2.7.1+cu118"
assert torch.cuda.is_available(), "PyTorch cannot see NVIDIA CUDA"
assert wp.is_cuda_available(), "Warp cannot see NVIDIA CUDA"
print("GPU PREFLIGHT PASS:", torch.cuda.get_device_name(0))
PY
python scripts/maintenance/verify_source_manifests.py --check
python scripts/maintenance/check_project_policy.py --scope full
python -m pytest -q tests/ research/zhang_xue/tests/
```

Nếu cần xác nhận chất lượng ở quy mô E4096, chạy CUDA gates và các regression bổ sung riêng. Kiểm tra package **không** tương đương với hoàn thành Phase 1 hay hội tụ MATD3/MASAC.

### Ma trận đã thực sự chứng minh

| Gói | V124 reference | Tình trạng |
| --- | --- | --- |
| Python | 3.10.22 | đã chạy trên Vast |
| torch | 2.7.1+cu118 | CUDA thực PASS trên Vast RTX 3060 |
| warp-lang | 1.18.0+cu12 | CUDA thực PASS; **khác với wheel mặc định CUDA 13** |
| numpy | 2.2.6 | đã có trong môi trường Vast đã qua test |
| PyYAML | 6.0.4 | đã có trong môi trường Vast đã qua test |
| SimPy | 4.1.2 | đã có trong môi trường Vast đã qua test |
| pytest | 9.2.0 | 644 passed / 6 skipped / 153 subtests trên nhánh port kernel |
| setuptools | 79.0.1 | đã có trong môi trường Vast |
| tomli | 2.5.0 | đã có trong Vast Python 3.10; dùng khi không có tomllib |
| matplotlib | 3.10.7 | tùy chọn, **chưa được kiểm thử trong gate GPU trước đó** |

Hạn chế: CUDA wheel +cu12 chỉ dành cho **Linux x86_64**. Driver thấp hơn R525 không tương thích với yêu cầu Warp CUDA12. PyTorch CUDA 11.8 và Warp CUDA12 hoạt động đồng thời trên môi trường RTX 3060 đã thử nghiệm, nhưng **không mặc định bảo đảm trên GPU Blackwell, các ảnh Vast khác, driver khác**. Trên các máy đó, phải thực hiện preflight, test và hiệu chỉnh riêng; không tự động thay Torch/Warp.

Mã dự án không import `pandas`, `scipy`, `torchvision`, `torchaudio`, `wandb` hay `huggingface_hub` trên đường chạy cần thiết; không đưa vào bộ tối thiểu để tránh làm nặng image. Vẽ đồ thị là phần tùy chọn.

Nguồn: `pyproject.toml`, `research/zhang_xue/pyproject.toml`, `setup.sh`, `scripts/maintenance/install_warp_compat.py`, môi trường Vast thực tế 28/11/2026; PyTorch official previous versions và NVIDIA Warp 1.18 release.
