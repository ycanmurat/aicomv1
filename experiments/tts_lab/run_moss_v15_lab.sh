#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_ROOT="$PROJECT_ROOT/.runtime/voice-lab/moss-v15-flagship"
PYTHON="$RUNTIME_ROOT/.venv/bin/python"
PUBLIC_PORT="${MOSS_LAB_PORT:-18083}"

if [[ ! -x "$PYTHON" || ! -x "$RUNTIME_ROOT/bin/crispasr" ]]; then
  echo "Run ./experiments/tts_lab/setup_moss_v15_flagship.sh first." >&2
  exit 1
fi

# Seed the isolated voice directory from the synthetic reference already owned
# by this local workspace. The server copies it and never modifies the source.
DEFAULT_REFERENCE="$PROJECT_ROOT/../aicall/data/audio/voice-reference/ada.wav"
if [[ -z "${MOSS_FLAGSHIP_REFERENCE:-}" && -f "$DEFAULT_REFERENCE" ]]; then
  export MOSS_FLAGSHIP_REFERENCE="$DEFAULT_REFERENCE"
fi

echo "Fatma flagship lab: http://127.0.0.1:$PUBLIC_PORT"
echo "The first start loads and warms MOSS-TTS v1.5 Qwen3-8B."
echo "Press Ctrl-C to stop only this lab and its owned model process."

cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT" exec "$PYTHON" -m uvicorn \
  experiments.tts_lab.flagship_server:app \
  --host 127.0.0.1 \
  --port "$PUBLIC_PORT" \
  --no-access-log
