#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
mode=${1:-core}

case "$mode" in
    core|full) ;;
    *) echo "Usage: $0 [core|full]" >&2; exit 2 ;;
esac

cd "$project_dir"
command -v uv >/dev/null 2>&1 || { echo "uv was not found." >&2; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg was not found." >&2; exit 1; }
command -v ollama >/dev/null 2>&1 || { echo "Ollama was not found." >&2; exit 1; }
command -v whisper-cli >/dev/null 2>&1 || { echo "whisper-cli was not found." >&2; exit 1; }

uv python install 3.11
uv sync --extra dev --python 3.11

if [ ! -f .env ]; then
    cp .env.example .env
fi

if ! curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    echo "Start the Ollama service, then run this script again." >&2
    exit 1
fi
ollama pull qwen3.5:9b

whisper_model="$project_dir/models/ggml-large-v3-turbo-q8_0.bin"
if [ ! -f "$whisper_model" ]; then
    echo "Downloading Whisper large-v3-turbo q8 weights..."
    curl -fL --retry 3 \
        -o "$whisper_model.part" \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin"
    mv "$whisper_model.part" "$whisper_model"
fi

if [ "$mode" = full ]; then
    installer=$(mktemp "${TMPDIR:-/tmp}/nemo-speech-install.XXXXXX")
    trap 'rm -f "$installer"' EXIT HUP INT TERM
    curl -fsSL \
        "https://github.com/NVIDIA/NeMo-Speech.cpp/raw/main/scripts/install.sh" \
        -o "$installer"
    sh "$installer" \
        --prefix "$project_dir/.runtime/nemo-speech" \
        --backend metal \
        --no-modify-path
    NEMO_SPEECH_MODEL_DIR="$project_dir/models/nemo-cache" \
        "$project_dir/.runtime/nemo-speech/bin/nemo-speech" pull nemotron-3.5

    if [ ! -d "$project_dir/.runtime/FreyaTTS/.git" ]; then
        git clone --depth 1 https://github.com/freyavoiceai/FreyaTTS.git \
            "$project_dir/.runtime/FreyaTTS"
    else
        git -C "$project_dir/.runtime/FreyaTTS" pull --ff-only
    fi
    uv sync --extra dev --extra freya --python 3.11
    HF_HOME="$project_dir/models/huggingface" HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
        uv run --no-sync python -c \
        "import sys; sys.path.insert(0, sys.argv[1]); from freyatts import FreyaTTS; FreyaTTS.from_pretrained('freyavoice/freya-tts', device='cpu'); print('FreyaTTS is ready.')" \
        "$project_dir/.runtime/FreyaTTS"
fi

uv run --offline --no-sync aicom-doctor || true
echo "Start AICOM: ./scripts/run.sh"
