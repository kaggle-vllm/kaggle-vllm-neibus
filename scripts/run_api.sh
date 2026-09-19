#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
kaggle-vllm-nebius serve --host 127.0.0.1 --port 8000 --reload
