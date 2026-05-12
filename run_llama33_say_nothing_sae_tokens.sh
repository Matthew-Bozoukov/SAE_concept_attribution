#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

if [[ -z "${NEURONPEDIA_API_KEY:-}" ]]; then
  echo "Error: set NEURONPEDIA_API_KEY before running this experiment." >&2
  exit 1
fi

exec uv run python llama33_say_nothing_sae_tokens.py --direction both "$@"
