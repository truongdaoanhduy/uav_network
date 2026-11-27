#!/usr/bin/env python3
"""Generate real, reproducible, bounded QA logs under gitignored outputs/.

This never downloads datasets, checks accounts, or treats BLOCKED GPU as PASS.
Logs and JUnit XML may be copied into a source-release ZIP but not Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def run_gate(name: str, argv: list[str], output_dir: Path, *, cwd: Path = ROOT,
             blocked_codes: tuple[int, ...] = (), timeout: int = 180) -> dict:
    """Capture exact stdout/stderr and exit code, including intentional blockers."""
    if not LOG_NAME.fullmatch(name):
        raise ValueError("Invalid gate name")
    log = output_dir / "logs" / f"{name}.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(argv, cwd=cwd, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
        status = ("PASS" if result.returncode == 0 else
                  "BLOCKED" if result.returncode in blocked_codes else "FAIL")
        log.write_text(result.stdout, encoding="utf-8")
        code = result.returncode
    except subprocess.TimeoutExpired as error:
        status, code = "TIMEOUT", None
        body = error.stdout or b""
        if isinstance(body, bytes):
            body = body.decode("utf-8", errors="replace")
        log.write_text(body + f"\nTIMEOUT_AFTER_SECONDS={timeout}\n", encoding="utf-8")
    return {"name": name, "argv": argv, "cwd": str(cwd),
            "status": status, "return_code": code,
            "stdout_file": str(log.relative_to(output_dir)), "timeout_s": timeout}


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment_info(seed: int) -> dict:
    torch_version = _package_version("torch")
    cuda_available = False
    if torch_version:
        import torch
        cuda_available = bool(torch.cuda.is_available())
    source = ROOT / "SOURCE_SHA256_MANIFEST.json"
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "platform": platform.platform(),
            "python_executable": sys.executable,
            "torch": torch_version, "warp_lang": _package_version("warp-lang"),
            "pytest": _package_version("pytest"), "cuda_available": cuda_available,
            "seed": seed, "run_type": "CPU regression + GPU readiness only; NOT training",
            "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest()
            if source.exists() else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qa_evidence"))
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()):
        parser.error("QA output must be inside gitignored outputs/")
    output.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    os.environ["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT), str(ROOT / "research/zhang_xue/src"),
         os.environ.get("PYTHONPATH", "")))
    junit = output / "pytest_junit.xml"
    gate_argv = [
        ("pytest_all", [python, "-m", "pytest", "-q", "-ra",
                         "--junitxml=" + str(junit), "tests/", "research/zhang_xue/tests/"]),
        ("source_manifest", [python, "scripts/maintenance/verify_source_manifests.py", "--check"]),
        ("project_policy", [python, "scripts/maintenance/check_project_policy.py", "--scope", "full"]),
        ("map_direct_link", [python, "scripts/maintenance/academic_scenario_audit.py", "--sizes-m", "1000", "2000", "3000"]),
        ("paired_scenario_profiles", [python, "scripts/maintenance/paired_scenario_gate.py"]),
        ("memory_budget_estimates", [python, "scripts/maintenance/network_memory_report.py",
                                      "--output", str(output / "memory_estimates.json")]),
        ("warp_cpu_packet_execution", [python, "scripts/gpu/warp_packet_cuda_gate.py",
                                       "--envs", "4", "--device", "cpu"]),
        ("cpu_warp_application_contract", [python, "scripts/gpu/packet_contract_gate.py",
                                          "--uavs", "6", "--device", "cpu"]),
        ("scientific_inventory", [python, "scripts/maintenance/v72_scientific_inventory.py",
                                   "--output-dir", str(output), "--seed", str(args.seed)]),
        ("scenario_2km_smoke", [python, "research/zhang_xue/scripts/run_scenario.py", "--steps", "2", "--seed", str(args.seed),
                     "--config", "research/zhang_xue/configs/scenario_v67_datamule_2km.yaml"]),
        ("scenario_3km_stress_smoke", [python, "research/zhang_xue/scripts/run_scenario.py", "--steps", "1", "--seed", str(args.seed),
                     "--config", "research/zhang_xue/configs/scenario_v70_datamule_3km.yaml"]),
        ("cpu_matd3", [python, "research/zhang_xue/scripts/v61_cpu_actor_learning_gate.py",
                       "--algorithm", "matd3", "--seed", str(args.seed), "--transitions", "4",
                       "--learner-config", "configs/learners/matd3_cpu_gate.yaml",
                       "--scenario-config", "research/zhang_xue/configs/scenario_v67_datamule_1km.yaml"]),
        ("cpu_masac", [python, "research/zhang_xue/scripts/v61_cpu_actor_learning_gate.py",
                       "--algorithm", "masac", "--seed", str(args.seed), "--transitions", "4",
                       "--learner-config", "configs/learners/masac_cpu_gate.yaml",
                       "--scenario-config", "research/zhang_xue/configs/scenario_v67_datamule_1km.yaml"]),
        ("gpu_readiness", [python, "scripts/maintenance/q1_gpu_preflight.py"]),
        ("cuda_packet_execution", [python, "scripts/gpu/warp_packet_cuda_gate.py", "--envs", "4"]),
    ]
    results = [run_gate(name, cmd, output, blocked_codes=(3,) if name in ("gpu_readiness", "cuda_packet_execution", "warp_cpu_packet_execution", "cpu_warp_application_contract") else ())
               for name, cmd in gate_argv]
    summary = {"environment": environment_info(args.seed), "gates": results,
               "overall": "FAIL" if any(r["status"] in ("FAIL", "TIMEOUT") for r in results)
               else "CPU_PASS_GPU_BLOCKED" if any(r["status"] == "BLOCKED" for r in results)
               else "PASS"}
    (output / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Reproducible QA evidence", "", "These files are **command outputs**, not handwritten PASS claims.",
             "", f"Overall: **{summary['overall']}**", "", "| Gate | Status | Exit code | Full log |",
             "|---|---|---:|---|"]
    for r in results:
        lines.append(f"| {r['name']} | {r['status']} | {r['return_code']} | `{r['stdout_file']}` |")
    lines += ["", "`pytest_junit.xml` records individual test cases.",
              "The GPU readiness check is explicitly **BLOCKED** unless all packet-event prerequisites pass.",
              "A four-transition learner smoke is **not** convergence training.", ""]
    (output / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"overall": summary["overall"],
                      "gates": {r["name"]: r["status"] for r in results},
                      "output_dir": str(output)}, indent=2))
    return 2 if summary["overall"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
