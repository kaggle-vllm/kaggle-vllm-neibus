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

Run controlled TP=1/TP=2 or serving-concurrency benchmarks using the existing
`kaggle-vllm` commands. Preserve the raw upstream artifact unchanged, then
create a normalized EvidenceBundle beside it.

Do not claim that Nemotron 3.5 Lightning runs in the Kaggle T4 runtime. In this
project Nemotron is served separately through Nebius Token Factory.
