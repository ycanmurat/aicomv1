#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAB_ROOT="$PROJECT_ROOT/.runtime/voice-lab"
MOSS_ROOT="$LAB_ROOT/moss-tts-nano"
VENV_ROOT="$LAB_ROOT/.venv"
MOSS_REPOSITORY="https://github.com/OpenMOSS/MOSS-TTS-Nano.git"

mkdir -p "$LAB_ROOT/output"

if [[ ! -d "$MOSS_ROOT/.git" ]]; then
  git clone --depth 1 "$MOSS_REPOSITORY" "$MOSS_ROOT"
fi

if [[ ! -x "$VENV_ROOT/bin/python" ]]; then
  uv venv --python 3.12 "$VENV_ROOT"
fi

uv pip install --python "$VENV_ROOT/bin/python" -e "$MOSS_ROOT" huggingface-hub soundfile

echo "MOSS-TTS-Nano deney ortamı hazır."
echo "Başlatmak için: ./experiments/tts_lab/run_moss.sh"
