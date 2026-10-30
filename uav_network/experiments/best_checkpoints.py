"""Select immutable actors on validation; retain exact last training state separately."""
from contextlib import suppress
import hashlib
import io
import json
import math
from pathlib import Path

import torch

from ..configuration import Config, _apply, config_fingerprint
from .comparison_metrics import evaluation_protocol
from ..configuration_xyz import strip_legacy_xyz_defaults

CRITERIA = ("success_rate", "mean_delivery_rate_percent", "mean_search_rate_percent", "mean_return")


def selection_identity(config):
    d = strip_legacy_xyz_defaults(config.to_dict())
    value = {k:d[k] for k in ("world", "sensor", "network", "energy", "reward", config.runtime.algorithm, "pql")}
    value["runtime"] = {k:d["runtime"][k] for k in
                        ("algorithm", "seed", "amp_mode", "num_envs", "environment_backend")}
    value["validation"] = [config.tracking.periodic_evaluation_seed,
                           config.tracking.periodic_evaluation_episodes]
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _config_hashes(config):
    legacy = strip_legacy_xyz_defaults(config.to_dict())
    return {config_fingerprint(config),
            hashlib.sha256(json.dumps(legacy,sort_keys=True,allow_nan=False).encode()).hexdigest()}


def _validation_seeds(config):
    first = config.tracking.periodic_evaluation_seed
    last = first + config.tracking.periodic_evaluation_episodes - 1
    test_first = config.runtime.eval_seed
    test_last = test_first + config.runtime.eval_episodes - 1
    if first <= test_last and test_first <= last:
        raise ValueError("Validation and final test banks overlap")
    return list(range(first, last+1))


def _valid_score(score):
    return (isinstance(score, list) and len(score)==4
            and all(type(v) in (int,float) and math.isfinite(v) for v in score)
            and 0<=score[0]<=1 and 0<=score[1]<=100 and 0<=score[2]<=100)


def _snapshot(data):
    saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if saved.get("purpose") != "evaluation_only" or saved.get("version") != 1:
        raise ValueError("Best selection requires an evaluation-only actor snapshot")
    c = Config()
    _apply(c, saved["config"])
    c.validate()
    if not saved.get("actors") or any(not torch.isfinite(v).all() for v in saved["actors"].values()):
        raise ValueError("Best actors are nonfinite or empty")
    return saved, c


class BestCheckpoint:
    def __init__(self, config, output, state=None):
        self.config, self.output = config, Path(output)
        self.path = self.output / "best.pt"
        self.metadata = self.data = None
        self.identity = selection_identity(config)
        if state is not None:
            _validation_seeds(config)
            values = state.get("checkpoint_bytes")
            if not isinstance(values, torch.Tensor) or values.dtype != torch.uint8 or values.ndim != 1:
                raise ValueError("Invalid best-checkpoint bytes")
            data = values.cpu().numpy().tobytes()
            meta = state["metadata"]
            if not isinstance(meta, dict) or hashlib.sha256(data).hexdigest() != meta.get("checkpoint_sha256"):
                raise ValueError("Best-checkpoint bytes changed")
            saved, before = _snapshot(data)
            if (meta.get("identity") != self.identity or selection_identity(before) != self.identity
                    or hashlib.sha256(data).hexdigest() != meta.get("checkpoint_sha256")
                    or meta.get("config_sha256") not in _config_hashes(before)
                    or meta.get("global_step") != saved["progress"]["global_step"]
                    or meta.get("episode") != saved["progress"]["episodes"]
                    or meta.get("kind") != "validation_best"
                    or meta.get("validation_seeds") != list(range(config.tracking.periodic_evaluation_seed,
                        config.tracking.periodic_evaluation_seed + config.tracking.periodic_evaluation_episodes))
                    or meta.get("policy_version") != saved["source"]["policy_version"]
                    or meta.get("criteria") != list(CRITERIA)
                    or not _valid_score(meta.get("score"))
                    or not isinstance(meta.get("metrics"), dict)
                    or not _valid_score([meta["metrics"].get(k) for k in CRITERIA])
                    or meta["score"] != [meta["metrics"].get(k) for k in CRITERIA]
                    or set(meta["metrics"]) != set(CRITERIA)
                    or meta.get("scenario_hash") != evaluation_protocol(before, [], {})["scenario_hash"]):
                raise ValueError("Best-checkpoint state identity/provenance differs")
            self._publish(data, meta)
            self.data, self.metadata = data, meta

    def _publish(self, data, meta):
        self.output.mkdir(parents=True, exist_ok=True)
        model_tmp = self.path.with_suffix(".pt.tmp")
        meta_path = self.output / "best.json"
        meta_tmp = meta_path.with_suffix(".json.tmp")
        try:
            # Prepare both complete files before replacing either published file.
            model_tmp.write_bytes(data)
            meta_tmp.write_text(json.dumps(meta, indent=2, allow_nan=False))
            model_tmp.replace(self.path)
            meta_tmp.replace(meta_path)
        finally:
            for path in (model_tmp, meta_tmp):
                with suppress(OSError):
                    path.unlink(missing_ok=True)

    def consider(self, result, source):
        if source.get("kind") != "periodic_validation":
            return False
        _validation_seeds(self.config)
        data = Path(source["checkpoint"]).read_bytes()
        saved, c = _snapshot(data)
        digest = hashlib.sha256(data).hexdigest()
        protocol, policy = result.get("protocol", {}), result.get("policy_provenance", {})
        seed = self.config.tracking.periodic_evaluation_seed
        n = self.config.tracking.periodic_evaluation_episodes
        seeds = list(range(seed, seed+n))
        score = [result.get(k) for k in CRITERIA]
        if (selection_identity(c) != self.identity or digest != source["checkpoint_sha256"]
                or policy.get("checkpoint_sha256") != digest
                or policy.get("config_sha256") != config_fingerprint(c)
                or policy.get("algorithm") != c.runtime.algorithm
                or policy.get("training_mode") != ("pql" if c.pql.enabled else "base")
                or policy.get("amp_mode") != c.runtime.amp_mode
                or result.get("algorithm") != c.runtime.algorithm
                or result.get("episodes") != n or source.get("evaluation_episodes") != n
                or source.get("seed_first") != seed or source.get("seed_last") != seeds[-1]
                or [r.get("seed") for r in result.get("records", [])] != seeds
                or protocol.get("seeds") != seeds or protocol.get("deterministic_policy") is not True
                or protocol.get("training_seed_overlap") is not False
                or protocol.get("training_seed") != c.runtime.seed
                or protocol.get("scenario_hash") != evaluation_protocol(c, [], {})["scenario_hash"]
                or protocol.get("training_global_step") != source["global_step"]
                or saved["progress"]["global_step"] != source["global_step"]
                or saved["progress"]["episodes"] != source["episodes"]
                or saved["source"]["policy_version"] != source["policy_version"]
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in score)):
            raise ValueError("Best candidate policy/validation provenance differs")
        if not (0 <= score[0] <= 1 and 0 <= score[1] <= 100 and 0 <= score[2] <= 100):
            raise ValueError("Best candidate rates outside their units")
        if self.metadata is not None and tuple(score) <= tuple(self.metadata["score"]):
            return False
        meta = {"kind":"validation_best", "identity":self.identity, "criteria":list(CRITERIA),
                "score":score, "metrics":{k:result[k] for k in CRITERIA},
                "checkpoint_sha256":digest, "config_sha256":config_fingerprint(c),
                "global_step":source["global_step"], "episode":source["episodes"],
                "policy_version":source["policy_version"], "validation_seeds":seeds,
                "scenario_hash":protocol["scenario_hash"]}
        self._publish(data, meta)
        self.data, self.metadata = data, meta
        return True

    def state_dict(self):
        if self.metadata is None:
            return None
        return {"metadata":self.metadata,
                "checkpoint_bytes":torch.frombuffer(bytearray(self.data), dtype=torch.uint8).clone()}

    def result(self, last_path):
        if self.metadata is None:
            return {"checkpoint":str(last_path),
                    "checkpoint_selection":{"kind":"unvalidated_final_fallback",
                                            "reason":"No valid periodic validation completed"}}
        # Re-publish the in-memory committed winner if a prior I/O error interrupted a pair.
        try:
            valid = (hashlib.sha256(self.path.read_bytes()).hexdigest() == self.metadata["checkpoint_sha256"]
                     and json.loads((self.output / "best.json").read_text()) == self.metadata)
        except (OSError, ValueError):
            valid = False
        if not valid:
            self._publish(self.data, self.metadata)
        return {"checkpoint":str(self.path), "checkpoint_selection":dict(self.metadata)}
