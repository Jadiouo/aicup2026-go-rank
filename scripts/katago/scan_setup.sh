#!/bin/bash
# Prepare pinned KataGo assets from original archives or Kaggle's extracted layout.
# Usage: scan_setup.sh <ROOT> [ASSET_DIR ...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:?usage: scan_setup.sh ROOT [ASSET_DIR...]}"; shift
PY="${PYTHON:-python3}"
K="$ROOT/katago"
mkdir -p "$K"
pin() { "$PY" - "$HERE/scan_pins.json" "$1" "$2" <<'P'
import json,sys
print(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])
P
}
verify() {
  local got
  got="$(sha256sum "$1" | cut -d' ' -f1)"
  [ "$got" = "$2" ] || { echo "SHA256 MISMATCH $1: $got != $2" >&2; exit 1; }
  echo "verified $1 $got"
}
prepare_net() {
  local key="$1" name archive raw d
  shift
  name="$(pin "$key" name)"; archive="$K/$name"
  raw="$K/${name%.gz}/model.bin"
  mkdir -p "$(dirname "$raw")"
  if [ ! -f "$archive" ] && [ ! -f "$raw" ]; then
    for d in "$@"; do
      if [ -f "$d/$name" ]; then cp "$d/$name" "$archive"; break; fi
      if [ -f "$d/${name%.gz}/model.bin" ]; then
        cp "$d/${name%.gz}/model.bin" "$raw"; break
      fi
    done
  fi
  if [ -f "$archive" ]; then
    verify "$archive" "$(pin "$key" sha256)"
    [ -f "$raw" ] || gzip -dc "$archive" > "$raw"
  elif [ ! -f "$raw" ]; then
    echo "missing $name in mounted assets; downloading pinned archive" >&2
    wget -q "$(pin "$key" url)" -O "$archive"
    verify "$archive" "$(pin "$key" sha256)"
    gzip -dc "$archive" > "$raw"
  fi
  verify "$raw" "$(pin "$key" content_sha256)"
}
prepare_engine() {
  local name archive d
  name="$(pin katago_zip name)"; archive="$K/$name"
  if [ ! -f "$archive" ] && [ ! -f "$K/katago" ]; then
    for d in "$@"; do
      if [ -f "$d/$name" ]; then cp "$d/$name" "$archive"; break; fi
      if [ -f "$d/${name%.zip}/katago" ]; then
        cp "$d/${name%.zip}/katago" "$K/katago"; break
      fi
    done
  fi
  if [ -f "$archive" ]; then
    verify "$archive" "$(pin katago_zip sha256)"
    [ -f "$K/katago" ] || (cd "$K" && unzip -oq "$archive" katago)
  elif [ ! -f "$K/katago" ]; then
    echo "missing $name in mounted assets; downloading pinned archive" >&2
    wget -q "$(pin katago_zip url)" -O "$archive"
    verify "$archive" "$(pin katago_zip sha256)"
    (cd "$K" && unzip -oq "$archive" katago)
  fi
  verify "$K/katago" "$(pin katago_zip executable_sha256)"
  chmod +x "$K/katago"
}
prepare_engine "$@"
prepare_net main_net "$@"
prepare_net human_net "$@"
[ -d "$ROOT/cu12/nvidia" ] || pip install -q --upgrade --target "$ROOT/cu12" $("$PY" -c "import json;print(' '.join(json.load(open('$HERE/scan_pins.json'))['cuda_pip']))")
export LD_LIBRARY_PATH="$(ls -d "$ROOT"/cu12/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
ldd_output="$(ldd "$K/katago" 2>&1 || true)"  # static/odd binaries make ldd exit 1; only "not found" is fatal
if [[ "$ldd_output" == *"not found"* ]]; then
  echo "$ldd_output" >&2
  exit 1
fi
"$K/katago" version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
