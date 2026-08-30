#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_ROOT="$PROJECT_ROOT/.runtime/voice-lab/moss-v15-flagship"
CRISPASR="$RUNTIME_ROOT/bin/crispasr"
MODEL="$RUNTIME_ROOT/cache/moss-tts-v1.5-q4_k.gguf"
CODEC="$RUNTIME_ROOT/cache/moss-tts-v1.5-codec.gguf"
OUTPUT_DIR="$RUNTIME_ROOT/output"

TEXT="${*:-Merhaba, ben Fatma. Size nasıl yardımcı olabilirim?}"
OUTPUT="${MOSS_OUTPUT:-$OUTPUT_DIR/moss-v15-flagship-tr.wav}"
CPU_THREADS="${MOSS_CPU_THREADS:-4}"

for required_file in "$CRISPASR" "$MODEL" "$CODEC"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing runtime asset: $required_file" >&2
    echo "Run ./experiments/tts_lab/setup_moss_v15_flagship.sh first." >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR" "$(dirname "$OUTPUT")"

arguments=(
  --backend moss-tts
  -m "$MODEL"
  --codec-model "$CODEC"
  --gpu-backend metal
  -t "$CPU_THREADS"
  # CrispASR v0.8.30 ignores -tl for this backend. -l tr becomes the official
  # MOSS prompt value "Turkish".
  -l tr
  --speaker-identity synthetic
  --tts "$TEXT"
  --tts-output "$OUTPUT"
)

if [[ -n "${MOSS_REFERENCE_AUDIO:-}" ]]; then
  arguments+=(--voice "$MOSS_REFERENCE_AUDIO" --i-have-rights)
  if [[ "${MOSS_NO_SPOKEN_DISCLAIMER:-0}" == "1" ]]; then
    # The output still keeps C2PA provenance and the embedded watermark.
    arguments+=(--no-spoken-disclaimer --accept-marking-responsibility)
  fi
fi

CRISPASR_CACHE_DIR="$RUNTIME_ROOT/cache" \
  /usr/bin/time -l "$CRISPASR" "${arguments[@]}"

echo "Output: $OUTPUT"
