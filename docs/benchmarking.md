# Benchmarking

The submission should use a controlled matrix with identical model revision,
prompt corpus, input/output lengths, and runtime settings except for the one
factor under investigation.

Primary demonstration:

- TP=1 vs TP=2;
- concurrency 1 / 8 / 16 / 32 where practical;
- output throughput;
- request throughput;
- TTFT p50/p95;
- TPOT p50/p95;
- GPU utilization and VRAM where collected.

Avoid claiming a single root cause unless a dedicated experiment isolates it.

`compare_matrix` pairs TP=1 and TP=2 at each concurrency, validates model
identity, and calculates the first measured benefit rather than hardcoding a
crossover. For throughput metrics it also reports the diagnostic ratio:

```text
TP scaling efficiency = TP2 throughput / (2 × TP1 throughput)
```

This ratio is useful for this controlled comparison but is not presented as the
only universal scaling-efficiency definition.
