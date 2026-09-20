# Evidence schema

`EvidenceBundle` is the project boundary between raw GPU execution and model
reasoning.

Important rules:

1. Missing values stay `null`; do not estimate them.
2. Preserve the raw benchmark artifact separately.
3. Record environment and runtime provenance.
4. Do not turn causal hypotheses into measured fields.
5. Checksum submission-quality raw artifacts.
6. Record source schema, filename, repository, available commit, and capture time.
7. Preserve per-GPU telemetry; never sum per-device VRAM into a fake device limit.
8. Convert source seconds to milliseconds explicitly for TTFT/TPOT/latency fields.

The production adapter accepts `kaggle-vllm-serving-benchmark-v1`. Unsupported
schemas, non-executed results, missing required metrics, inconsistent
concurrency, and malformed hardware records fail loudly.
