#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$project_dir"

if [ ! -d .venv ]; then
    echo "Run ./scripts/bootstrap.sh first." >&2
    exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec uv run --offline --no-sync aicom "$@"
