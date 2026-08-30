#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAB_ROOT="$PROJECT_ROOT/.runtime/voice-lab"
MOSS_ROOT="$LAB_ROOT/moss-tts-nano"
VENV_ROOT="$LAB_ROOT/.venv"
MOSS_REPOSITORY="https://github.com/OpenMOSS/MOSS-TTS-Nano.git"
MOSS_REVISION="cc7bdf19c7639c0870dab22045a33b442760f6be"

export HF_HOME="$LAB_ROOT/cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export UV_CACHE_DIR="$LAB_ROOT/cache/uv"

mkdir -p "$LAB_ROOT/output"

if [[ ! -d "$MOSS_ROOT/.git" ]]; then
  git init "$MOSS_ROOT"
  git -C "$MOSS_ROOT" remote add origin "$MOSS_REPOSITORY"
  git -C "$MOSS_ROOT" fetch --depth 1 origin "$MOSS_REVISION"
  git -C "$MOSS_ROOT" checkout --detach FETCH_HEAD
fi

if [[ "$(git -C "$MOSS_ROOT" rev-parse HEAD)" != "$MOSS_REVISION" ]]; then
  echo "The existing lab checkout differs from the tested revision; it was not changed." >&2
  exit 1
fi

if [[ ! -x "$VENV_ROOT/bin/python" ]]; then
  uv venv --python 3.12 "$VENV_ROOT"
fi

uv pip install --python "$VENV_ROOT/bin/python" -e "$MOSS_ROOT" \
  -r "$PROJECT_ROOT/experiments/tts_lab/requirements.txt"
"$VENV_ROOT/bin/python" "$PROJECT_ROOT/experiments/tts_lab/prepare_models.py"

echo "MOSS-TTS-Nano lab is ready."
echo "Start: ./experiments/tts_lab/run_moss.sh"
