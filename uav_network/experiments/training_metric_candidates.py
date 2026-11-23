"""Training-only candidate *proposals*; never select the final best policy from these metrics.

Each completed episode is generated during learning, possibly under multiple policies.
Its score is *not* a deterministic evaluation of the checkpoint we capture.
The selector only asks the snapshot bank to retain a promising Actor for later
independent, fixed-seed GPU validation.
"""
from collections import deque
from math import isfinite
from statistics import fmean


class TrainingMetricCandidates:
    def __init__(self, window=256):
        if not isinstance(window, int) or window < 1:
            raise ValueError("Training candidate window must be positive")
        self.window = window
        self.records = deque(maxlen=window)
        self.best_score = None
        self.best_at = None
        self.improvements = 0

    def update(self, records, *, global_step, episodes):
        """Return a candidate score when the trailing *completed* episode window improves.

        This ranking is deliberately lexicographic, matching BestCheckpoint:
        Mission Success, GCS Delivery %, Search %, Mean Return.
        """
        for rec in records:
            s = float(rec["success"])
            delivery = float(rec["delivery_rate_percent"])
            search = float(rec["search_rate_percent"])
            ret = float(rec["return"])
            if (not all(isfinite(v) for v in (s, delivery, search, ret))
                    or not 0 <= s <= 1
                    or not 0 <= delivery <= 100
                    or not 0 <= search <= 100):
                raise ValueError("Invalid training episode metrics for snapshot candidate")
            self.records.append((s, delivery, search, ret))
        if len(self.records) < self.window:
            return None
        score = tuple(float(fmean(row[i] for row in self.records)) for i in range(4))
        if self.best_score is not None and score <= self.best_score:
            return None
        self.best_score = score
        self.best_at = {"global_step": int(global_step), "episodes": int(episodes)}
        self.improvements += 1
        return {
            "reason": "training_metric_best_candidate",
            "training_metric_window_episodes": self.window,
            "score": dict(zip(("success_rate", "mean_delivery_rate_percent",
                               "mean_search_rate_percent", "mean_return"), score)),
            "global_step": int(global_step),
            "episodes": int(episodes),
            "candidate_rank": self.improvements,
            "disclaimer": "Collected under learning/exploration; select best only with Validation."
        }
