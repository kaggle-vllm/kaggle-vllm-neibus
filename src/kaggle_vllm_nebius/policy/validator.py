from __future__ import annotations

from dataclasses import dataclass

from kaggle_vllm_nebius.evidence.schema import EvidenceBundle
from kaggle_vllm_nebius.recommendations.schema import ExperimentSpec

from .t4 import T4_ALLOWED_DTYPES, T4_ALLOWED_TP


@dataclass(frozen=True)
class PolicyResult:
    valid: bool
    errors: list[str]
    warnings: list[str]


def validate_experiment(
    experiment: ExperimentSpec,
    evidence: EvidenceBundle,
) -> PolicyResult:
    errors: list[str] = []
    warnings: list[str] = []

    gpu_names = {gpu.name.lower() for gpu in evidence.gpus}
    is_t4 = all("t4" in name for name in gpu_names)

    if experiment.tensor_parallel_size > len(evidence.gpus):
        errors.append("Requested TP exceeds the available GPU count.")

    if is_t4 and experiment.tensor_parallel_size not in T4_ALLOWED_TP:
        errors.append("Validated Kaggle T4 policy allows TP=1 or TP=2 only.")

    if is_t4 and experiment.dtype.lower() not in T4_ALLOWED_DTYPES:
        errors.append("Validated Kaggle T4 policy currently permits FP16 only.")

    if experiment.gpu_memory_utilization is not None and experiment.gpu_memory_utilization > 0.95:
        warnings.append("gpu_memory_utilization above 0.95 leaves very little headroom.")

    if is_t4:
        warnings.append(
            "Tesla T4 / SM75 does not support the H100-specific NVFP4/"
            "FlashInfer/DSpark recipe from the Nemotron reference notebook."
        )

    return PolicyResult(valid=not errors, errors=errors, warnings=warnings)
