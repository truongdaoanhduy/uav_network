#!/usr/bin/env python3
"""Install Warp 1.18 with the binary matching the observed NVIDIA driver.

Official release: https://github.com/NVIDIA/warp/releases/tag/v1.18.0
Only Linux x86_64 automatic installation is implemented. No driver is modified.
"""
from __future__ import annotations
import argparse
import platform
import subprocess
import sys

CU12_WHEEL = ("https://github.com/NVIDIA/warp/releases/download/v1.18.0/"
              "warp_lang-1.18.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl")


def requirement_for_driver(driver_version: str) -> str:
    try:
        major = int(driver_version.strip().split(".")[0])
    except (ValueError, TypeError):
        raise ValueError("an observed numeric NVIDIA driver version is required") from None
    if major < 525:
        raise ValueError("Warp CUDA12 wheel requires an R525-or-newer driver")
    return "warp-lang==1.18.0" if major >= 580 else CU12_WHEEL


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--driver-version",help="explicit observed driver, otherwise query nvidia-smi")
    ap.add_argument("--dry-run",action="store_true")
    a=ap.parse_args()
    if platform.system()!="Linux" or platform.machine() not in ("x86_64","AMD64"):
        ap.error("automatic installation supports Linux x86_64; consult the pinned official release for other platforms")
    driver=a.driver_version
    if driver is None:
        driver=subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True).splitlines()[0]
    req=requirement_for_driver(driver)
    print(f"Observed driver {driver}: selected {req}",flush=True)
    if not a.dry_run:
        subprocess.run([sys.executable,"-m","pip","install",req],check=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
