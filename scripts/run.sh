#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$project_dir"

if [ ! -d .venv ]; then
    echo "Önce ./scripts/bootstrap.sh çalıştırılmalı." >&2
    exit 1
fi

exec uv run aicom "$@"
