#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
kaggle-vllm-nebius token-factory-smoke
