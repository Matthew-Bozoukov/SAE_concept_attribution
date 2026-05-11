#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

exec .venv/bin/python run_llama31_70b_unfaithful_examples_vllm.py "$@"
