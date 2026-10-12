#!/usr/bin/env bash
# Fetch only UavNetSim runtime folders; synthesize scenarios at env.reset().
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
python_bin="${PYTHON:-python3}"
venv_flags=()
package_spec='.'
for option in "$@"; do
  case "$option" in
    # Kaggle's Python image lacks a working ensurepip; inherit its existing pip.
    --system-site-packages) venv_flags+=(--system-site-packages --without-pip) ;;
    --cloud) package_spec='.[cloud]' ;;
    *) echo 'Usage: bash setup.sh [--system-site-packages] [--cloud]' >&2; exit 2 ;;
  esac
done
"$python_bin" -c 'import sys; assert sys.version_info >= (3,11), "Python >=3.11 required"'
if [[ ! -x .venv/bin/python ]]; then
  "$python_bin" -m venv "${venv_flags[@]}" .venv
fi
.venv/bin/python -m pip --version
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e "$package_spec" -r scripts/requirements-network.txt
upstream_dir="$project_root/third_party/UavNetSim"
revision="$(tr -d '\r\n' < scripts/upstream-revision.txt)"
mkdir -p third_party outputs
if [[ ! -d "$upstream_dir/.git" ]]; then
  git init -q "$upstream_dir"
  git -C "$upstream_dir" remote add origin https://github.com/Zihao-Felix-Zhou/UavNetSim.git
fi
git -C "$upstream_dir" sparse-checkout init --cone
git -C "$upstream_dir" sparse-checkout set allocation energy entities mac mobility \
  path_planning phy routing scene simulator telemetry topology utils
if ! git -C "$upstream_dir" cat-file -e "$revision^{commit}" 2>/dev/null; then
  git -C "$upstream_dir" fetch --depth=1 --filter=blob:none origin "$revision"
fi
git -C "$upstream_dir" checkout --detach "$revision"
.venv/bin/python -c 'from uav_network.configuration import load_config; from uav_network.communication.upstream import load_upstream; from gymnasium.vector import AutoresetMode; load_upstream(load_config()); assert AutoresetMode.DISABLED; print("UavNetSim and Gymnasium process runtime ready")'
echo 'Ready. Run: .venv/bin/python -m uav_network --help'
