from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceCitation(StrictModel):
    claim: str
    evidence_paths: list[str] = Field(min_length=1)
    confidence: Literal["low", "medium", "high"]


class VerificationMetric(str, Enum):
    REQUEST_THROUGHPUT_RPS = "request_throughput_rps"
    OUTPUT_TOKENS_PER_SECOND = "output_tokens_per_second"
    TTFT_P50_MS = "ttft_p50_ms"
    TTFT_P95_MS = "ttft_p95_ms"
    TPOT_P50_MS = "tpot_p50_ms"
    TPOT_P95_MS = "tpot_p95_ms"
    AVG_GPU_UTILIZATION_PCT = "avg_gpu_utilization_pct"
    PEAK_GPU_MEMORY_MIB = "peak_gpu_memory_mib"


class MetricConstraint(StrictModel):
    metric: VerificationMetric
    operator: Literal["lt", "le", "gt", "ge"]
    value: float


class ExperimentSpec(StrictModel):
    tensor_parallel_size: int = Field(ge=1)
    concurrency: int = Field(ge=1)
    dtype: str = "float16"
    max_model_len: int | None = Field(default=None, gt=0)
    gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1)
    enforce_eager: bool | None = None
    disable_custom_all_reduce: bool | None = None
    objective: str = Field(min_length=1)
    rationale: str
    verification_metrics: list[VerificationMetric] = Field(min_length=1)
    constraints: list[MetricConstraint] = Field(default_factory=list)


class DiagnosisReport(StrictModel):
    summary: str
    findings: list[EvidenceCitation]
    inferences: list[EvidenceCitation] = Field(default_factory=list)
    insufficient_evidence: list[str] = Field(default_factory=list)
    recommendation: ExperimentSpec | None = None
    warnings: list[str] = Field(default_factory=list)


def validate_diagnosis_for_mode(
    report: DiagnosisReport,
    *,
    require_experiment: bool,
) -> DiagnosisReport:
    if require_experiment and report.recommendation is None and not report.insufficient_evidence:
        raise ValueError(
            "Optimization mode requires an ExperimentSpec or specific missing evidence."
        )
    return report
