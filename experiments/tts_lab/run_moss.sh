#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAB_ROOT="$PROJECT_ROOT/.runtime/voice-lab"
MOSS_ROOT="$LAB_ROOT/moss-tts-nano"
PYTHON="$LAB_ROOT/.venv/bin/python"
LAUNCHER="$PROJECT_ROOT/experiments/tts_lab/launch_moss.py"

if [[ ! -x "$PYTHON" || ! -f "$MOSS_ROOT/app_onnx.py" ]]; then
  echo "Önce ./experiments/tts_lab/setup_moss.sh komutunu çalıştırın." >&2
  exit 1
fi

mkdir -p "$LAB_ROOT/output"

exec "$PYTHON" "$LAUNCHER" \
  --host 127.0.0.1 \
  --port "${MOSS_PORT:-18083}" \
  --cpu-threads "${MOSS_CPU_THREADS:-4}"
