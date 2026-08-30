#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAB_ROOT="$PROJECT_ROOT/.runtime/voice-lab"
MOSS_ROOT="$LAB_ROOT/moss-tts-nano"
PYTHON="$LAB_ROOT/.venv/bin/python"
LAUNCHER="$PROJECT_ROOT/experiments/tts_lab/launch_moss.py"

export HF_HOME="$LAB_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1
export DO_NOT_TRACK=1

if [[ ! -x "$PYTHON" || ! -f "$MOSS_ROOT/app_onnx.py" ]]; then
  echo "Önce ./experiments/tts_lab/setup_moss.sh komutunu çalıştırın." >&2
  exit 1
fi

mkdir -p "$LAB_ROOT/output"

exec "$PYTHON" "$LAUNCHER" \
  --host 127.0.0.1 \
  --port "${MOSS_PORT:-18083}" \
  --cpu-threads "${MOSS_CPU_THREADS:-4}" "$@"
