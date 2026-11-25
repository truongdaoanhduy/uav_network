#!/usr/bin/env bash
# Pin exact classic SimPy/802.11b UavNetSim-v1 implementation, distinct from
# newer Zihao-Felix-Zhou/UavNetSim (Sionna RT). MIT license lives in clone.
set -euo pipefail
SHA=64bc3a2424f2976418731ee94f9a8fd3e8810b22
URL=https://github.com/EdwardS201/UavNetSim-v1.git
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$REPO_ROOT/outputs/reference_sources/UavNetSim-v1-classic}"
if [[ -e "$DEST" && ! -d "$DEST/.git" ]]; then
  echo "Refusing existing non-Git target: $DEST" >&2; exit 1
fi
FRESH=0
if [[ ! -d "$DEST/.git" ]]; then
  FRESH=1
  mkdir -p "$(dirname "$DEST")"
  git clone --filter=blob:none --no-checkout "$URL" "$DEST"
fi
if ! git -C "$DEST" cat-file -e "$SHA^{commit}" 2>/dev/null; then
  git -C "$DEST" fetch --depth=1 origin "$SHA"
fi
if [[ "$FRESH" == 0 && -n "$(git -C "$DEST" status --porcelain)" ]]; then
  echo "Refusing to overwrite a dirty vendor checkout at $DEST" >&2; exit 1
fi
git -C "$DEST" checkout --detach "$SHA"
ACTUAL="$(git -C "$DEST" rev-parse HEAD)"
if [[ "$ACTUAL" != "$SHA" ]]; then echo "Wrong vendor SHA" >&2;exit 1;fi
echo "UAV_CLASSIC_V1_ROOT=$DEST"
echo "PINNED_SHA=$ACTUAL"
