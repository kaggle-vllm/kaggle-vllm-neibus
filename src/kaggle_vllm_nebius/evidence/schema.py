from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Provenance(StrictModel):
    project: str
    project_version: str
    git_commit: str | None = None
    timestamp: datetime | None = None


class GPUInfo(StrictModel):
    index: int = Field(ge=0)
    name: str
    compute_capability: str | None = None
    memory_mib: int = Field(gt=0)


class EnvironmentInfo(StrictModel):
    platform: str
    python_version: str | None = None
    pytorch_version: str | None = None
    cuda_version: str | None = None
    vllm_version: str | None = None
    driver_version: str | None = None


class ModelInfo(StrictModel):
    model_id: str
    revision: str | None = None
    dtype: str | None = None


class RuntimeInfo(StrictModel):
    tensor_parallel_size: int = Field(ge=1)
    attention_backend: str | None = None
    enforce_eager: bool | None = None
    disable_custom_all_reduce: bool | None = None
    max_model_len: int | None = Field(default=None, gt=0)
    gpu_memory_utilization: float | None = Field(default=None, gt=0.0, le=1.0)


class WorkloadInfo(StrictModel):
    concurrency: int = Field(ge=1)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    request_count: int | None = Field(default=None, ge=1)


class PerformanceMetrics(StrictModel):
    request_throughput_rps: float | None = Field(default=None, ge=0)
    output_tokens_per_second: float | None = Field(default=None, ge=0)
    ttft_p50_ms: float | None = Field(default=None, ge=0)
    ttft_p95_ms: float | None = Field(default=None, ge=0)
    tpot_p50_ms: float | None = Field(default=None, ge=0)
    tpot_p95_ms: float | None = Field(default=None, ge=0)


class GPUMetrics(StrictModel):
    avg_utilization_pct: float | None = Field(default=None, ge=0, le=100)
    peak_memory_mib: float | None = Field(default=None, ge=0)
    avg_power_watts: float | None = Field(default=None, ge=0)


class EvidenceBundle(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    provenance: Provenance
    environment: EnvironmentInfo
    gpus: list[GPUInfo] = Field(min_length=1)
    model: ModelInfo
    runtime: RuntimeInfo
    workload: WorkloadInfo
    metrics: PerformanceMetrics
    gpu_metrics: GPUMetrics = Field(default_factory=GPUMetrics)
    errors: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def topology_is_plausible(self) -> EvidenceBundle:
        if self.runtime.tensor_parallel_size > len(self.gpus):
            raise ValueError(
                "tensor_parallel_size cannot exceed the number of GPUs recorded "
                "in the evidence bundle"
            )
        return self
