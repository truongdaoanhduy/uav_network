"""Small structured run evidence; logging failures never become learner failures."""
from contextlib import suppress
from datetime import datetime, timezone
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path

READING_GUIDE = """### Đọc dashboard đúng cách
- **Chất lượng nhiệm vụ:** delivery và success cao hơn trên cùng scenario, seed và budget; return chỉ là bằng chứng bổ sung. So sánh cùng replay/precision; không gộp trung bình các thuật toán khác nhau.
- **Completed outcomes** là episode đã kết thúc trong cửa sổ transition. Xem cohort/episodes, cohort/partial và source_start/end_step; live-world search/battery phụ thuộc tuổi các episode đang chạy.
- **Fixed-policy evaluation** dùng checkpoint và seed cố định; **Final CPU** hiển thị từng episode đánh giá. Ô trống là thiếu dữ liệu. Profile chính hiện tắt mini-eval định kỳ; không suy ra generalization chỉ từ training return.
- Loss/TD/Q/gradient là chẩn đoán, không có ngưỡng loss chung để chấm tốt/xấu. PDR là packet, khác delivery mục tiêu tới GCS. Energy/delivery chỉ có nghĩa khi delivery/success tương đương.
- GPU/CPU/RAM và throughput giải thích tốc độ, không chấm chất lượng policy. Sample-use BASE đo segment hiện tại; PQL dùng replay counters có thể đã restore.
- Mỗi plot tối đa 12 run riêng; bảng vẫn giữ mọi trạng thái. Chọn các run cùng cấu hình để so sánh. Created là lúc W&B tạo run; Runtime là phiên hiện tại, không phải tổng lineage.
- Logs/Files: training-events.jsonl, training-status.json, diagnostics.jsonl, cohorts.jsonl, failure.json. Logs chỉ ra triệu chứng và mốc xảy ra; traceback/batch/provenance mới giúp xác minh nguyên nhân.
"""

IDENTITY_FIELDS = {"run_id", "algorithm", "mode", "seed", "source_commit", "stage",
                   "account", "num_envs", "batch_size", "replay_cpu_fraction", "root_kernel_ref"}
EVIDENCE_METRICS = {
    "learner/actor_loss", "learner/critic_loss", "learner/td_error_abs_mean",
    "learner/q_abs_max", "learner/critic_grad_norm", "learner/actor_grad_norm",
    "learner/continuous_entropy", "learner/discrete_entropy",
    "cohort/return_mean", "cohort/delivery_rate_percent_mean", "cohort/search_rate_percent_mean",
    "cohort/success_mean", "cohort/episodes", "cohort/partial",
    "cohort/source_start_step", "cohort/source_end_step", "diagnostics/warning_count",
    "diagnostics/fatal_error_count", "recovery/status", "recovery/attempts",
    "recovery/checkpoint_bytes", "recovery/elapsed_seconds", "recovery/error_type",
    "evaluation_status/status", "evaluation_status/completed", "evaluation_status/failed",
    "evaluation/episodes_completed", "evaluation/return", "evaluation/search_rate_percent",
    "evaluation/delivery_rate_percent", "evaluation/success", "evaluation/checkpoint_global_step",
    "eval/mean_delivery_rate_percent", "eval/success_rate", "eval/mean_return",
}


class RunJournal:
    def __init__(self, output, identity):
        self.output = Path(output)
        self.secrets = [value for key, value in os.environ.items()
                        if (key in {"WANDB_API_KEY", "KAGGLE_API_TOKEN", "HF_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"}
                            or key.startswith("KAGGLE_API_TOKEN_ACCOUNT_")) and len(value) >= 8]
        self.identity = {k: self._scalar(v) for k, v in identity.items() if k in IDENTITY_FIELDS}
        self.context, self.warned = {}, False
        try:
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / "training-events.jsonl").touch()
            (self.output / "metrics-guide.md").write_text(READING_GUIDE)
        except OSError as error:
            self._warn(error)

    def _scalar(self, value):
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, Real):
            return float(value) if math.isfinite(value) else ("NaN" if math.isnan(value) else str(value))
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, "[REDACTED]")
            return value[:256]
        return "<" + type(value).__name__ + ">"

    def _warn(self, error):
        if not self.warned:
            self.warned = True
            with suppress(OSError):
                print("Optional run journal I/O failed: " + type(error).__name__, flush=True)

    def record(self, event, values=None):
        self.context.update({k: self._scalar(v) for k, v in (values or {}).items()
                             if not any(word in k.lower() for word in ("api_key", "password", "secret", "token"))})
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": event,
                  "identity": self.identity, "context": self.context.copy()}
        encoded = json.dumps(record, allow_nan=False, sort_keys=True)
        with suppress(OSError):
            print(json.dumps({"training_event": record}, allow_nan=False), flush=True)
        try:
            with (self.output / "training-events.jsonl").open("a") as stream:
                stream.write(encoded + "\n")
        except OSError as error:
            self._warn(error)
        try:
            temporary = self.output / "training-status.json.tmp"
            temporary.write_text(encoded + "\n")
            temporary.replace(self.output / "training-status.json")
        except OSError as error:
            self._warn(error)
        return record
