# Kaggle execution adapter

The local package is intentionally CUDA-independent.

On Kaggle, install the existing runtime and this hackathon package, then convert
the real benchmark output into the `EvidenceBundle` schema.

Recommended sequence:

```bash
python -m pip install "kaggle-vllm[hub]==0.2.0"
kaggle-vllm bootstrap --strict
eval "$(kaggle-vllm env)"
kaggle-vllm doctor --strict
```

Run controlled TP=1/TP=2 serving-concurrency benchmarks using the existing
`kaggle-vllm` commands. Import the resulting artifact without editing it:

```bash
kaggle-vllm-nebius import-kaggle-run \
  --input /path/to/raw-serving.json \
  --output artifacts/public/normalized/run.json
```

The importer validates `kaggle-vllm-serving-benchmark-v1`, preserves the raw
bytes, records SHA256 and available provenance, normalizes seconds to
milliseconds where required, and writes a typed `EvidenceBundle`.

`normalize_kaggle_run.py` is retained only for development compatibility with
older hand-entered examples. Manual transcription is not the production path.

Do not claim that Nemotron 3.5 Lightning runs in the Kaggle T4 runtime. In this
project Nemotron is served separately through Nebius Token Factory.
