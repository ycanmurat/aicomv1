#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_ROOT="$PROJECT_ROOT/.runtime/voice-lab/moss-v15-flagship"
BIN_DIR="$RUNTIME_ROOT/bin"
CACHE_DIR="$RUNTIME_ROOT/cache"
DOWNLOAD_DIR="$RUNTIME_ROOT/downloads"
VENV_DIR="$RUNTIME_ROOT/.venv"
REQUIREMENTS="$PROJECT_ROOT/experiments/tts_lab/flagship_requirements.txt"

CRISP_VERSION="0.8.30"
CRISP_ARCHIVE="crispasr-macos-v${CRISP_VERSION}.tar.gz"
CRISP_SHA256="51b83d6b4a2d68e0a7d1f5351963ac162b250b3f2fc0791813db4928b88e096c"
CRISP_URL="https://github.com/CrispStrobe/CrispASR/releases/download/v${CRISP_VERSION}/crispasr-macos.tar.gz"

# This is the official MOSS-TTS-v1.5 8B model converted to GGUF Q4_K by cstr.
# The architecture remains Qwen3-8B; this is not the separate 4B Local model.
MODEL_REVISION="75f875ec3cd46cbfa0942a881ec3af85d2ed9b40"
MODEL_REPOSITORY="https://huggingface.co/cstr/moss-tts-v1.5-GGUF"
MODEL_FILE="moss-tts-v1.5-q4_k.gguf"
MODEL_SHA256="9e7fb9ed28339be5327dce16f9bd53c67220cbf119a9c767f513d28d1fa80547"
CODEC_FILE="moss-tts-v1.5-codec.gguf"
CODEC_SHA256="9cb5aa788dfb8fb0d46323898fb8dba57fb85d9c1f615019949c7f00a4bd5382"

for command_name in curl shasum tar uv; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
done

mkdir -p "$BIN_DIR" "$CACHE_DIR" "$DOWNLOAD_DIR"

sha256_of() {
  shasum -a 256 "$1" | awk '{print $1}'
}

verify_file() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256_of "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "Checksum mismatch: $path" >&2
    echo "Expected: $expected" >&2
    echo "Actual:   $actual" >&2
    return 1
  fi
}

download_verified() {
  local url="$1"
  local destination="$2"
  local expected_sha="$3"
  local partial="${destination}.download"

  if [[ -f "$destination" ]]; then
    verify_file "$destination" "$expected_sha"
    echo "Verified: $destination"
    return
  fi

  curl --fail --location --retry 3 --continue-at - \
    --output "$partial" "$url"
  verify_file "$partial" "$expected_sha"
  mv "$partial" "$destination"
  echo "Downloaded and verified: $destination"
}

download_verified "$CRISP_URL" "$DOWNLOAD_DIR/$CRISP_ARCHIVE" "$CRISP_SHA256"

if [[ ! -x "$BIN_DIR/crispasr" ]] || ! "$BIN_DIR/crispasr" --version 2>&1 | grep -q "version       : $CRISP_VERSION"; then
  extract_dir="$(mktemp -d)"
  trap 'rm -rf "$extract_dir"' EXIT
  tar -xzf "$DOWNLOAD_DIR/$CRISP_ARCHIVE" -C "$extract_dir"
  install -m 755 "$extract_dir/crispasr-macos/crispasr" "$BIN_DIR/crispasr"
  install -m 755 "$extract_dir/crispasr-macos/crispasr-quantize" "$BIN_DIR/crispasr-quantize"
  install -m 644 "$extract_dir/crispasr-macos/libc2pa_c.dylib" "$BIN_DIR/libc2pa_c.dylib"
  install -m 644 "$extract_dir/crispasr-macos/LICENSE" "$BIN_DIR/LICENSE"
  install -m 644 "$extract_dir/crispasr-macos/THIRD_PARTY_NOTICES.txt" "$BIN_DIR/THIRD_PARTY_NOTICES.txt"
fi

download_verified \
  "$MODEL_REPOSITORY/resolve/$MODEL_REVISION/$MODEL_FILE?download=true" \
  "$CACHE_DIR/$MODEL_FILE" \
  "$MODEL_SHA256"
download_verified \
  "$MODEL_REPOSITORY/resolve/$MODEL_REVISION/$CODEC_FILE?download=true" \
  "$CACHE_DIR/$CODEC_FILE" \
  "$CODEC_SHA256"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  uv venv --python 3.12 "$VENV_DIR"
fi
uv pip sync --python "$VENV_DIR/bin/python" "$REQUIREMENTS"

echo
echo "MOSS-TTS-v1.5 flagship experiment is ready."
echo "Runtime: CrispASR v$CRISP_VERSION"
echo "Backbone: Qwen3-8B, Q4_K GGUF"
echo "Quality lab: ./experiments/tts_lab/run_moss_v15_lab.sh"
echo "Single WAV: ./experiments/tts_lab/run_moss_v15_flagship.sh"
