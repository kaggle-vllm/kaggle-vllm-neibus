from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from kaggle_vllm_nebius.evidence.loader import save_evidence
from kaggle_vllm_nebius.evidence.schema import (
    DistributionStats,
    EnvironmentInfo,
    EvidenceBundle,
    GPUDeviceMetrics,
    GPUInfo,
    GPUMetrics,
    ModelInfo,
    PerformanceMetrics,
    Provenance,
    RequestDistribution,
    RuntimeInfo,
    WorkloadInfo,
)

SOURCE_SCHEMA = "kaggle-vllm-serving-benchmark-v1"
SOURCE_REPOSITORY = "https://github.com/kaggle-vllm/kaggle-vllm"


class KaggleServingArtifactError(ValueError):
    """Raised when a raw serving artifact cannot be normalized safely."""


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise KaggleServingArtifactError(f"Missing or invalid object: {key}")
    return value


def _required(parent: dict[str, Any], key: str, expected_type: type | tuple[type, ...]):
    value = parent.get(key)
    if isinstance(value, bool) and expected_type in (int, float, (int, float)):
        raise KaggleServingArtifactError(f"Invalid field type: {key}")
    if not isinstance(value, expected_type):
        raise KaggleServingArtifactError(f"Missing or invalid field: {key}")
    return value


def _distribution(
    source: dict[str, Any] | None,
    *,
    scale: float = 1.0,
) -> DistributionStats | None:
    if not source:
        return None
    count = _required(source, "count", int)

    def scaled(name: str) -> float | None:
        value = source.get(name)
        return None if value is None else float(value) * scale

    return DistributionStats(
        count=count,
        minimum=scaled("min"),
        maximum=scaled("max"),
        mean=scaled("mean"),
        p50=scaled("p50"),
        p95=scaled("p95"),
        p99=scaled("p99"),
    )


def _cuda_release(hardware: dict[str, Any]) -> str | None:
    torch_cuda = hardware.get("torch_cuda")
    if isinstance(torch_cuda, str) and torch_cuda:
        return torch_cuda
    toolkit = hardware.get("cuda_toolkit")
    if isinstance(toolkit, str):
        match = re.search(r"release\s+([0-9.]+)", toolkit)
        if match:
            return match.group(1)
    return None


def _topology(raw: dict[str, Any]) -> tuple[str | None, bool | None]:
    topology = raw.get("topology")
    if not isinstance(topology, dict):
        return None, None
    parsed = topology.get("parsed_matrix")
    if not isinstance(parsed, dict):
        return None, None
    links = parsed.get("links")
    link = None
    if isinstance(links, list) and links and isinstance(links[0], dict):
        candidate = links[0].get("path")
        link = candidate if isinstance(candidate, str) else None
    observed = parsed.get("nvlink_observed")
    return link, observed if isinstance(observed, bool) else None


def _model_identity(engine: dict[str, Any]) -> tuple[str, str | None]:
    served_name = _required(engine, "served_model_name", str)
    source = engine.get("model")
    if not isinstance(source, str):
        return served_name, None
    match = re.search(r"models--([^/]+?)/snapshots/", source)
    if match:
        return match.group(1).replace("--", "/"), source
    return served_name, source


def normalize_kaggle_serving_artifact(
    raw: dict[str, Any],
    *,
    source_filename: str,
    source_sha256: str,
    source_commit: str | None = None,
) -> EvidenceBundle:
    schema = raw.get("schema_version")
    if schema != SOURCE_SCHEMA:
        raise KaggleServingArtifactError(
            f"Unsupported source schema {schema!r}; expected {SOURCE_SCHEMA!r}."
        )
    if raw.get("status") != "executed":
        raise KaggleServingArtifactError("Only successfully executed serving artifacts import.")

    engine = _mapping(raw, "engine")
    workload = _mapping(raw, "workload")
    measurements = _mapping(raw, "measurements")
    hardware = _mapping(raw, "hardware")
    identity = _mapping(raw, "identity")

    required_metrics = (
        "request_throughput_per_second",
        "output_throughput_tokens_per_second",
        "ttft_seconds",
        "tpot_seconds",
        "input_tokens",
        "output_tokens",
        "successful_requests",
        "failed_requests",
    )
    missing = [name for name in required_metrics if measurements.get(name) is None]
    if missing:
        raise KaggleServingArtifactError(
            "Required measured fields are missing: " + ", ".join(sorted(missing))
        )

    ttft = _mapping(measurements, "ttft_seconds")
    tpot = _mapping(measurements, "tpot_seconds")
    for name, distribution in (("ttft_seconds", ttft), ("tpot_seconds", tpot)):
        for percentile in ("p50", "p95"):
            if distribution.get(percentile) is None:
                raise KaggleServingArtifactError(
                    f"Required measured field is missing: {name}.{percentile}"
                )

    raw_gpus = hardware.get("gpus")
    if not isinstance(raw_gpus, list) or not raw_gpus:
        raise KaggleServingArtifactError("hardware.gpus must contain at least one GPU.")
    gpus: list[GPUInfo] = []
    for item in raw_gpus:
        if not isinstance(item, dict):
            raise KaggleServingArtifactError("hardware.gpus contains a non-object entry.")
        capability = item.get("capability")
        compute_capability = None
        if (
            isinstance(capability, list)
            and len(capability) == 2
            and all(isinstance(part, int) for part in capability)
        ):
            compute_capability = f"{capability[0]}.{capability[1]}"
        total_memory = _required(item, "total_memory", int)
        gpus.append(
            GPUInfo(
                index=_required(item, "index", int),
                name=_required(item, "name", str),
                compute_capability=compute_capability,
                memory_mib=round(total_memory / (1024 * 1024)),
            )
        )

    telemetry = raw.get("gpu_telemetry")
    telemetry = telemetry if isinstance(telemetry, dict) else {}
    summaries = telemetry.get("summaries")
    summaries = summaries if isinstance(summaries, list) else []
    per_gpu: list[GPUDeviceMetrics] = []
    for item in summaries:
        if not isinstance(item, dict):
            continue
        per_gpu.append(
            GPUDeviceMetrics(
                index=_required(item, "index", int),
                sample_count=_required(item, "sample_count", int),
                avg_utilization_pct=item.get("mean_utilization_percent"),
                peak_utilization_pct=item.get("peak_utilization_percent"),
                peak_memory_mib=item.get("peak_memory_used_mib"),
                peak_power_watts=item.get("peak_power_draw_w"),
            )
        )

    server = raw.get("server")
    server = server if isinstance(server, dict) else {}
    visible = server.get("visible_physical_gpu_indices")
    active_indices = set(visible) if isinstance(visible, list) else {item.index for item in per_gpu}
    active = [item for item in per_gpu if item.index in active_indices]
    utilization_values = [
        item.avg_utilization_pct for item in active if item.avg_utilization_pct is not None
    ]
    memory_values = [item.peak_memory_mib for item in active if item.peak_memory_mib is not None]

    native_runtime = identity.get("native_runtime")
    native_runtime = native_runtime if isinstance(native_runtime, dict) else {}
    topology_link, nvlink_observed = _topology(raw)

    captured_at = identity.get("captured_at_utc")
    raw_source_commit = identity.get("source_git_commit")
    source_project_commit = raw_source_commit if isinstance(raw_source_commit, str) else None
    failure_observations = raw.get("failure_observations")
    errors = (
        [str(item) for item in failure_observations]
        if isinstance(failure_observations, list)
        else []
    )

    model_name, model_source = _model_identity(engine)
    served_name = _required(engine, "served_model_name", str)
    revision = engine.get("model_revision")
    dtype = engine.get("dtype")
    concurrency = _required(workload, "concurrency", int)
    if raw.get("concurrency") != concurrency:
        raise KaggleServingArtifactError("Top-level and workload concurrency do not match.")

    return EvidenceBundle(
        run_id=Path(source_filename).stem,
        provenance=Provenance(
            project="kaggle-vllm",
            project_version=_required(identity, "kaggle_vllm_version", str),
            git_commit=source_project_commit,
            timestamp=captured_at,
            source_schema=SOURCE_SCHEMA,
            source_filename=source_filename,
            source_sha256=source_sha256,
            source_repository=SOURCE_REPOSITORY,
            source_commit=source_commit,
            source_created_at=captured_at,
        ),
        environment=EnvironmentInfo(
            platform=_required(hardware, "platform", str),
            python_version=hardware.get("python"),
            pytorch_version=hardware.get("torch"),
            cuda_version=_cuda_release(hardware),
            vllm_version=identity.get("vllm_distribution_version"),
            driver_version=hardware.get("driver_version"),
            nccl_version=hardware.get("nccl"),
            runtime_wheel=native_runtime.get("wheel"),
            runtime_wheel_sha256=native_runtime.get("sha256"),
            runtime_revision=native_runtime.get("hf_revision"),
            topology_link=topology_link,
            nvlink_observed=nvlink_observed,
        ),
        gpus=gpus,
        model=ModelInfo(
            model_id=model_name,
            revision=revision if isinstance(revision, str) else None,
            dtype=dtype if isinstance(dtype, str) else None,
            served_name=served_name,
            source=model_source,
        ),
        runtime=RuntimeInfo(
            tensor_parallel_size=_required(engine, "tensor_parallel_size", int),
            attention_backend=None,
            enforce_eager=engine.get("enforce_eager"),
            disable_custom_all_reduce=engine.get("disable_custom_all_reduce"),
            max_model_len=engine.get("max_model_len"),
            gpu_memory_utilization=engine.get("gpu_memory_utilization"),
        ),
        workload=WorkloadInfo(
            concurrency=concurrency,
            input_tokens=_required(measurements, "input_tokens", int),
            output_tokens=_required(measurements, "output_tokens", int),
            request_count=_required(workload, "total_requests", int),
            successful_requests=_required(measurements, "successful_requests", int),
            failed_requests=_required(measurements, "failed_requests", int),
        ),
        metrics=PerformanceMetrics(
            request_throughput_rps=float(measurements["request_throughput_per_second"]),
            output_tokens_per_second=float(measurements["output_throughput_tokens_per_second"]),
            ttft_p50_ms=float(ttft["p50"]) * 1000.0,
            ttft_p95_ms=float(ttft["p95"]) * 1000.0,
            tpot_p50_ms=float(tpot["p50"]) * 1000.0,
            tpot_p95_ms=float(tpot["p95"]) * 1000.0,
        ),
        gpu_metrics=GPUMetrics(
            avg_utilization_pct=(
                sum(utilization_values) / len(utilization_values) if utilization_values else None
            ),
            peak_memory_mib=max(memory_values) if memory_values else None,
            per_gpu=per_gpu,
        ),
        request_distribution=RequestDistribution(
            input_tokens=_distribution(measurements.get("input_tokens_per_request")),
            output_tokens=_distribution(measurements.get("output_tokens_per_request")),
            latency_ms=_distribution(measurements.get("latency_seconds"), scale=1000.0),
        ),
        errors=errors,
        notes=[str(item) for item in raw.get("limitations", [])],
    )


def import_kaggle_serving_artifact(
    input_path: str | Path,
    output_path: str | Path,
    *,
    raw_directory: str | Path | None = None,
    source_commit: str | None = None,
) -> EvidenceBundle:
    source = Path(input_path)
    output = Path(output_path)
    raw_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise KaggleServingArtifactError(f"Invalid JSON in {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise KaggleServingArtifactError("Serving artifact root must be a JSON object.")

    if raw_directory is None:
        raw_base = output.parent.parent if output.parent.name == "normalized" else output.parent
        raw_directory = raw_base / "raw"
    raw_destination = Path(raw_directory) / source.name
    raw_destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != raw_destination.resolve():
        shutil.copy2(source, raw_destination)
    if hashlib.sha256(raw_destination.read_bytes()).hexdigest() != source_sha256:
        raise KaggleServingArtifactError("Preserved raw artifact checksum mismatch.")

    bundle = normalize_kaggle_serving_artifact(
        raw,
        source_filename=source.name,
        source_sha256=source_sha256,
        source_commit=source_commit,
    )
    save_evidence(bundle, output)
    return bundle
