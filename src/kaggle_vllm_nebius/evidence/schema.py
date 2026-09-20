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
    source_schema: str | None = None
    source_filename: str | None = None
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_repository: str | None = None
    source_commit: str | None = None
    source_created_at: datetime | None = None


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
    nccl_version: str | None = None
    runtime_wheel: str | None = None
    runtime_wheel_sha256: str | None = None
    runtime_revision: str | None = None
    topology_link: str | None = None
    nvlink_observed: bool | None = None


class ModelInfo(StrictModel):
    model_id: str
    revision: str | None = None
    dtype: str | None = None
    served_name: str | None = None
    source: str | None = None


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
    successful_requests: int | None = Field(default=None, ge=0)
    failed_requests: int | None = Field(default=None, ge=0)


class DistributionStats(StrictModel):
    count: int = Field(ge=0)
    minimum: float | None = Field(default=None, ge=0)
    maximum: float | None = Field(default=None, ge=0)
    mean: float | None = Field(default=None, ge=0)
    p50: float | None = Field(default=None, ge=0)
    p95: float | None = Field(default=None, ge=0)
    p99: float | None = Field(default=None, ge=0)


class RequestDistribution(StrictModel):
    input_tokens: DistributionStats | None = None
    output_tokens: DistributionStats | None = None
    latency_ms: DistributionStats | None = None


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
    per_gpu: list[GPUDeviceMetrics] = Field(default_factory=list)


class GPUDeviceMetrics(StrictModel):
    index: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    avg_utilization_pct: float | None = Field(default=None, ge=0, le=100)
    peak_utilization_pct: float | None = Field(default=None, ge=0, le=100)
    peak_memory_mib: float | None = Field(default=None, ge=0)
    peak_power_watts: float | None = Field(default=None, ge=0)


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
    request_distribution: RequestDistribution | None = None
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
