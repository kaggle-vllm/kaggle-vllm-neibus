# Public evidence catalog

These files are **real measured evidence**, not synthetic fixtures.

They were copied without modification from the pre-existing `kaggle-vllm`
repository's committed Milestone 2 serving matrix at source repository commit
`6ae0298fe0982e5a67a35e104868395e9cb3005d`.

- Source schema: `kaggle-vllm-serving-benchmark-v1`
- Model: `Qwen/Qwen2.5-3B-Instruct`
- Model revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- Hardware: two NVIDIA Tesla T4 GPUs; each run records its visible devices
- Matrix: TP=1/2 × concurrency 1/8/16/32
- `raw/`: exact source bytes
- `normalized/`: typed `EvidenceBundle` outputs from the production adapter
- `SHA256SUMS.txt`: checksums for both representations

The deterministic output-throughput analysis observes the first TP=2 advantage at
concurrency 16 for this matrix. It does not establish a universal crossover or a
single causal explanation.

The files under `artifacts/demo/` are separate synthetic/illustrative fixtures and
must not be described as benchmark measurements.
