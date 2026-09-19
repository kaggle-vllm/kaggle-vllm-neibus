# Kaggle-vLLM Inference Doctor

Measure → Diagnose → Recommend → Validate → Verify.

`kaggle-vllm-nebius` is an evidence-driven NVIDIA GPU inference engineering
agent for the Nebius x NVIDIA Global AI Hackathon.

The project combines:

- `kaggle-vllm` as the measured execution plane on Kaggle dual NVIDIA T4 GPUs;
- NVIDIA Nemotron 3.5 Lightning through Nebius Token Factory as the reasoning plane;
- typed evidence and recommendation schemas;
- read-only diagnostic tools;
- deterministic compatibility policy checks;
- audit trails for every diagnosis.

The original `kaggle-vllm` project is a pre-existing dependency. This repository
contains the new hackathon-period agent, evidence model, policy layer, API, and
validation workflow.

## Local quick start

This project is tested as a Python 3.11 application layer. It does not require
CUDA or vLLM for local development.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"

cp .env.example .env
kaggle-vllm-nebius doctor
pytest -q
```

To make a real Nebius Token Factory call, add your Builder Program API key:

```bash
nano .env
# NEBIUS_API_KEY=...
kaggle-vllm-nebius token-factory-smoke
```

To diagnose the included real-shaped demo evidence:

```bash
kaggle-vllm-nebius compare \
  artifacts/demo/tp1-c8.json \
  artifacts/demo/tp2-c8.json

kaggle-vllm-nebius diagnose \
  --baseline artifacts/demo/tp1-c8.json \
  --candidate artifacts/demo/tp2-c8.json \
  --question "Why did TP=2 underperform TP=1?"
```

`diagnose` requires `NEBIUS_API_KEY`. `compare`, `validate`, `doctor`, and the
test suite work fully offline.

## Architecture

```text
Kaggle dual T4
    │
    │ kaggle-vllm benchmark / telemetry evidence
    ▼
EvidenceBundle JSON
    │
    ├──────── deterministic comparison
    │
    ▼
Nemotron 3.5 Lightning
Nebius Token Factory
    │
    │ read-only tool calls
    ▼
DiagnosisReport
    │
    ▼
T4 compatibility policy
    │
    ▼
ExperimentSpec
    │
    ▼
Kaggle rerun → measured verification
```

## Safety model

The LLM never receives unrestricted shell execution. Nemotron may request
read-only evidence tools and returns a typed recommendation. A deterministic
policy layer validates the proposed vLLM settings before they can be rendered
as an execution plan.

## Repository relationship

- Core runtime: https://github.com/kaggle-vllm/kaggle-vllm
- Hackathon project: https://github.com/kaggle-vllm/kaggle-vllm-nebius
- Nebius cookbook: https://github.com/nebius/token-factory-cookbook
- NVIDIA Nemotron: https://github.com/NVIDIA-NeMo/Nemotron

See `docs/hackathon-changes.md` for the required existing-project disclosure.
