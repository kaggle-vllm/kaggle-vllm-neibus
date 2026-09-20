# Kaggle-vLLM Inference Doctor

Measure → Diagnose → Recommend → Validate → Rerun → Verify.

Kaggle-vLLM Inference Doctor is an evidence-driven NVIDIA GPU inference
engineering agent for the Nebius x NVIDIA Global AI Hackathon. It imports real
`kaggle-vllm` serving artifacts, computes comparisons and TP crossover facts in
Python, asks NVIDIA Nemotron to interpret those facts through read-only tools,
validates a typed experiment against hardware policy, and evaluates the measured
follow-up as `PASS`, `FAIL`, or `INCONCLUSIVE`.

The intended submission category is **Best Apps and Agents Track**.

## Architecture and project boundary

```text
kaggle-vllm (pre-existing measured runtime/research plane)
    ↓ raw checksummed serving evidence
kaggle-vllm-nebius adapter + SQLite evidence store
    ↓ deterministic comparison, crossover, and scaling analysis
Nebius Token Factory → nvidia/Nemotron-3_5-Lightning
    ↓ evidence-backed interpretation through read-only tools
typed ExperimentSpec → deterministic T4 policy
    ↓ controlled external rerun
VerificationReport (PASS / FAIL / INCONCLUSIVE)
```

`kaggle-vllm` remains the pre-existing CUDA execution dependency. This repository
contains the hackathon-period application. It does not claim that Nebius H100/H200
GPUs are native Kaggle accelerators. A future Nebius GPU execution backend is a
stretch goal and is not required by the working evidence loop.

Python calculates throughput/latency deltas, GPU deltas, TP scaling efficiency,
crossover points, constraints, and verification outcomes. Nemotron explains
plausible causes, evidence, trade-offs, missing evidence, and the next controlled
experiment. The model is never allowed to invent benchmark values.

## Local setup

Python 3.11 is required. CUDA and vLLM are not required for the application layer.

```bash
./scripts/bootstrap_local.sh
source .venv/bin/activate
kaggle-vllm-nebius doctor
```

The bootstrap reuses a healthy `.venv`. Offline validation never requires a
Nebius key:

```bash
ruff check src tests
pytest -q
python -m compileall -q src
```

## Import real Kaggle evidence

The production adapter accepts only `kaggle-vllm-serving-benchmark-v1`, preserves
the raw bytes, calculates SHA256, normalizes units, and rejects missing required
measurements.

```bash
kaggle-vllm-nebius import-kaggle-run \
  --input /path/to/qwen-tp1-c08.json \
  --output artifacts/public/normalized/qwen-tp1-c08.json \
  --source-commit <commit-containing-the-source-artifact>
```

When the output directory is named `normalized`, the immutable input is copied
beside it under `raw/`. The legacy `kaggle/normalize_kaggle_run.py` script is
development-only compatibility code; manual metric transcription is not the
production path.

## Deterministic matrix analysis

The repository includes eight real measured Qwen 2.5 3B artifacts: TP=1/2 at
concurrency 1/8/16/32. They are labeled and checksummed in
`artifacts/public/README.md`.

```bash
kaggle-vllm-nebius matrix artifacts/public/normalized/*.json \
  --metric output_tokens_per_second \
  --concurrency 1 8 16 32
```

For this exact measured matrix, TP=2 first exceeds TP=1 output-token throughput at
concurrency 16. This is a workload-specific observation, not a universal TP claim
or a causal explanation. TP scaling efficiency is reported as
`TP2 throughput / (2 × TP1 throughput)` as one diagnostic ratio, not the only
possible definition.

## Nebius Token Factory diagnosis

Set the key only in the ignored local `.env` file. The deterministic smoke call
keeps Nemotron reasoning disabled to require an exact sentinel response.

```bash
cp .env.example .env
# Add NEBIUS_API_KEY locally; never commit it.
kaggle-vllm-nebius token-factory-smoke

kaggle-vllm-nebius diagnose \
  --runs artifacts/public/normalized/*.json \
  --require-experiment \
  --question "Explain the measured TP behavior and propose a controlled next run."
```

The runtime integration uses Nebius Token Factory's OpenAI-compatible endpoint and
the NVIDIA open-source model `nvidia/Nemotron-3_5-Lightning`. Tool access is
read-only; no arbitrary shell is exposed.

## API and persistence

```bash
kaggle-vllm-nebius serve --host 127.0.0.1 --port 8000
```

The FastAPI service persists runs, diagnosis sessions, recommendations, and
verification reports in SQLite at `artifacts/private/kaggle-vllm-nebius.db` by
default. Set `KAGGLE_VLLM_NEBIUS_DB` to choose another local path. Unknown run IDs
return 404, invalid request models return 422, an unconfigured Token Factory returns
503, and diagnosis failures return 502. API responses do not expose private audit
traces or secrets.

## What was built during the submission period

After August 26, 2026, this repository added the evidence adapter/store,
deterministic comparison/crossover/verification layers, Nebius Token Factory
Nemotron agent, typed recommendations, T4 policy, audit trail, CLI, FastAPI service,
SQLite persistence, offline tests, and public checksummed evidence packaging. See
`docs/hackathon-changes.md` for the explicit pre-existing-project disclosure.

## Current limitations and roadmap

- The application does not itself execute CUDA workloads; measured reruns remain
  external controlled `kaggle-vllm` work.
- The included M2 matrix is real, but a new recommendation-specific rerun has not
  yet been claimed as verified by this hackathon application.
- Sampled GPU telemetry can miss instantaneous peaks.
- Nemotron interpretations can be wrong; deterministic policy and measured
  verification remain mandatory.
- A public hosted demo URL and a public three-minute YouTube demo are still
  submission deliverables.
- A future `ExecutionBackend` abstraction may add Nebius AI Cloud GPUs after the
  evidence loop is complete; no Nebius GPU execution is fabricated here.

## References

- Core runtime: https://github.com/kaggle-vllm/kaggle-vllm
- Hackathon application: https://github.com/kaggle-vllm/kaggle-vllm-nebius
- Nebius Token Factory cookbook: https://github.com/nebius/token-factory-cookbook
- NVIDIA Nemotron: https://github.com/NVIDIA-NeMo/Nemotron
- License: Apache-2.0 (`LICENSE`)
