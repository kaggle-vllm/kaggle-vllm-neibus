from __future__ import annotations

import shlex

from kaggle_vllm_nebius.recommendations.schema import ExperimentSpec


def render_benchmark_command(
    model: str,
    output_path: str,
    experiment: ExperimentSpec,
) -> str:
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
