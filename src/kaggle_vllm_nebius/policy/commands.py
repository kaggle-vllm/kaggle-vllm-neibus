from __future__ import annotations

import shlex

from kaggle_vllm_nebius.evidence.schema import EvidenceBundle
from kaggle_vllm_nebius.policy.validator import validate_experiment
from kaggle_vllm_nebius.recommendations.schema import ExperimentSpec


def render_benchmark_command(
    model: str,
    output_path: str,
    experiment: ExperimentSpec,
    evidence: EvidenceBundle,
) -> str:
    policy = validate_experiment(experiment, evidence)
    if not policy.valid:
        raise ValueError("Experiment rejected by policy: " + "; ".join(policy.errors))

    parts = [
        "kaggle-vllm",
        "benchmark-serving",
        "--model",
        model,
        "--tensor-parallel-size",
        str(experiment.tensor_parallel_size),
        "--concurrency",
        str(experiment.concurrency),
        "--output",
        output_path,
    ]

    if experiment.max_model_len is not None:
        parts += ["--max-model-len", str(experiment.max_model_len)]

    return " ".join(shlex.quote(part) for part in parts)
