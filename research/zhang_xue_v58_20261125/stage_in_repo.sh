#!/usr/bin/env bash
set -euo pipefail
# Stage a v56 research candidate onto a fresh NEW branch; leave v53 unchanged.
# Usage: bash stage_in_repo.sh /path/to/clean/v53/checkout
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${1:?Provide the clean v5.3 git checkout}"
BASE="experiment/zhang-xue-v53-audited-battery-contract-20261125"
CANDIDATE="experiment/zhang-xue-v56-sevenradio-gate3-20261125"
[[ -d "$REPO/.git" ]] || { echo >&2 'Not an ordinary .git directory'; exit 2; }
[[ "$(git -C "$REPO" branch --show-current)" == "$BASE" ]] || { echo >&2 "Wrong branch; expected $BASE"; exit 2; }
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || { echo >&2 'Dirty checkout, refusing edit'; exit 2; }
if git -C "$REPO" show-ref --quiet "refs/heads/$CANDIDATE"; then
    echo >&2 "Branch already exists: $CANDIDATE";exit 2
fi
for f in "$HERE"/src/*.py "$HERE"/tests/test_v54*.py "$HERE"/tests/test_v55*.py "$HERE"/tests/test_v56*.py "$HERE"/tests/v56_network_topology_cases.py; do
    case "$f" in
        "$HERE"/src/*) dest="$REPO/src/${f##*/}";;
        *) dest="$REPO/tests/${f##*/}";;
    esac
    [[ ! -e "$dest" ]] || { echo >&2 "Existing destination, refusing overwrite: $dest";exit 2; }
done
for f in "$HERE"/01_*.md "$HERE"/02_*.md "$HERE"/03_*.md "$HERE"/04_*.md;do
    [[ ! -e "$REPO/docs/superpowers/${f##*/}" ]] || { echo >&2 "Existing doc, refusing overwrite: $f";exit 2; }
done
[[ ! -e "$REPO/scripts/v56_cuda_preflight.py" ]] || { echo >&2 'Preflight already exists'; exit 2; }
# At this point all checks have passed; create *new* branch only.
git -C "$REPO" switch -c "$CANDIDATE"
mkdir -p "$REPO/src" "$REPO/tests" "$REPO/docs/superpowers" "$REPO/scripts"
cp "$HERE"/src/*.py "$REPO/src/"
cp "$HERE"/tests/test_v54*.py "$HERE"/tests/test_v55*.py "$HERE"/tests/test_v56*.py "$HERE"/tests/v56_network_topology_cases.py "$REPO/tests/"
cp "$HERE"/01_*.md "$HERE"/02_*.md "$HERE"/03_*.md "$HERE"/04_*.md "$REPO/docs/superpowers/"
cp "$HERE/scripts/v56_cuda_preflight.py" "$REPO/scripts/v56_cuda_preflight.py"
git -C "$REPO" add src tests scripts/v56_cuda_preflight.py docs/superpowers

git -C "$REPO" diff --cached --check
printf 'Staged v56 on NEW branch: %s\n' "$CANDIDATE"
printf 'Review with: git -C "%s" diff --cached --stat\n' "$REPO"
printf 'Then test with: PYTHONPATH=src /home/aduy/miniconda3/bin/python -m pytest -q tests/test_v54*.py tests/test_v55*.py tests/test_v56_warp_gate.py\n'
printf 'No commit/push/training performed automatically.\n'
