from __future__ import annotations

from dataclasses import dataclass

from .schema import EvidenceBundle


@dataclass(frozen=True)
class Delta:
    baseline: float | None
    candidate: float | None
    absolute: float | None
    percent: float | None


def _delta(baseline: float | None, candidate: float | None) -> Delta:
    if baseline is None or candidate is None:
        return Delta(baseline, candidate, None, None)
    absolute = candidate - baseline
    percent = None if baseline == 0 else (absolute / baseline) * 100.0
    return Delta(baseline, candidate, absolute, percent)


def compare_runs(
    baseline: EvidenceBundle,
    candidate: EvidenceBundle,
) -> dict:
    return {
        "baseline_run_id": baseline.run_id,
        "candidate_run_id": candidate.run_id,
        "same_model": baseline.model.model_id == candidate.model.model_id,
        "same_concurrency": baseline.workload.concurrency == candidate.workload.concurrency,
        "tensor_parallel": {
            "baseline": baseline.runtime.tensor_parallel_size,
            "candidate": candidate.runtime.tensor_parallel_size,
        },
        "output_tokens_per_second": _delta(
            baseline.metrics.output_tokens_per_second,
            candidate.metrics.output_tokens_per_second,
        ).__dict__,
        "request_throughput_rps": _delta(
            baseline.metrics.request_throughput_rps,
            candidate.metrics.request_throughput_rps,
        ).__dict__,
        "ttft_p95_ms": _delta(
            baseline.metrics.ttft_p95_ms,
            candidate.metrics.ttft_p95_ms,
        ).__dict__,
        "tpot_p95_ms": _delta(
            baseline.metrics.tpot_p95_ms,
            candidate.metrics.tpot_p95_ms,
        ).__dict__,
        "gpu_utilization_pct": _delta(
            baseline.gpu_metrics.avg_utilization_pct,
            candidate.gpu_metrics.avg_utilization_pct,
        ).__dict__,
    }
