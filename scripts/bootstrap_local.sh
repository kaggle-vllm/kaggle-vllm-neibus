#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3.11}"

echo "[1/5] Checking Python..."
"$PYTHON_BIN" --version
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'

echo "[2/5] Checking virtual environment..."
if [[ -x .venv/bin/python ]] && .venv/bin/python -c \
  'import sys; assert sys.version_info[:2] == (3, 11)' >/dev/null 2>&1; then
  echo "Reusing healthy Python 3.11 environment at .venv."
else
  if [[ -e .venv ]]; then
    echo "ERROR: .venv exists but is not a healthy Python 3.11 environment." >&2
    echo "Move it aside explicitly before rerunning bootstrap." >&2
    exit 1
  fi
  "$PYTHON_BIN" -m venv .venv
fi

echo "[3/5] Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[4/5] Installing project..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

echo "[5/5] Running offline acceptance checks..."
kaggle-vllm-nebius doctor
pytest -q
ruff check src tests

echo
echo "Local bootstrap complete."
echo "Activate later with: source .venv/bin/activate"
echo "Then add NEBIUS_API_KEY to .env for real Token Factory calls."
