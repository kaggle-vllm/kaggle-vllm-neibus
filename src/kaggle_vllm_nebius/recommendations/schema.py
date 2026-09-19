from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceCitation(StrictModel):
    claim: str
    evidence_paths: list[str] = Field(min_length=1)
    confidence: Literal["low", "medium", "high"]


class ExperimentSpec(StrictModel):
    tensor_parallel_size: int = Field(ge=1)
    concurrency: int = Field(ge=1)
    dtype: str = "float16"
    max_model_len: int | None = Field(default=None, gt=0)
    gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1)
    enforce_eager: bool | None = None
    disable_custom_all_reduce: bool | None = None
    rationale: str
    verification_metrics: list[str] = Field(default_factory=list)


class DiagnosisReport(StrictModel):
    summary: str
    findings: list[EvidenceCitation]
    recommendation: ExperimentSpec | None = None
    warnings: list[str] = Field(default_factory=list)
