from __future__ import annotations

from kaggle_vllm_nebius.evidence.comparison import compare_runs
from kaggle_vllm_nebius.evidence.schema import EvidenceBundle


def verify_experiment(
    baseline: EvidenceBundle,
    experiment: EvidenceBundle,
) -> dict:
    comparison = compare_runs(baseline, experiment)
    throughput = comparison["output_tokens_per_second"]
    improved = throughput["absolute"] is not None and throughput["absolute"] > 0
    return {
        "improved_output_throughput": improved,
        "comparison": comparison,
    }
