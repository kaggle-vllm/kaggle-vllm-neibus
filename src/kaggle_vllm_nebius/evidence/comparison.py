from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from kaggle_vllm_nebius.recommendations.schema import VerificationMetric

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


class MatrixPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concurrency: int
    tp1_run_id: str
    tp2_run_id: str
    tp1_value: float
    tp2_value: float
    delta_percent: float | None
    tp_scaling_efficiency: float | None
    tp2_beneficial: bool


class CrossoverAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: VerificationMetric
    crossover_observed: bool
    first_concurrency: int | None
    tp1_value: float | None
    tp2_value: float | None
    delta_percent: float | None
    points: list[MatrixPoint]
    scaling_efficiency_formula: str


_LOWER_IS_BETTER = {
    VerificationMetric.TTFT_P50_MS,
    VerificationMetric.TTFT_P95_MS,
    VerificationMetric.TPOT_P50_MS,
    VerificationMetric.TPOT_P95_MS,
    VerificationMetric.PEAK_GPU_MEMORY_MIB,
}


def metric_value(bundle: EvidenceBundle, metric: VerificationMetric) -> float | None:
    if metric == VerificationMetric.AVG_GPU_UTILIZATION_PCT:
        return bundle.gpu_metrics.avg_utilization_pct
    if metric == VerificationMetric.PEAK_GPU_MEMORY_MIB:
        return bundle.gpu_metrics.peak_memory_mib
    return getattr(bundle.metrics, metric.value)


def compare_matrix(
    runs: Iterable[EvidenceBundle],
    *,
    metric: VerificationMetric = VerificationMetric.OUTPUT_TOKENS_PER_SECOND,
    concurrencies: Iterable[int] | None = None,
) -> CrossoverAnalysis:
    grouped: dict[int, dict[int, EvidenceBundle]] = {}
    expected_concurrencies = set(concurrencies) if concurrencies is not None else None
    for run in runs:
        concurrency = run.workload.concurrency
        if expected_concurrencies is not None and concurrency not in expected_concurrencies:
            continue
        tp = run.runtime.tensor_parallel_size
        if tp not in {1, 2}:
            continue
        if tp in grouped.setdefault(concurrency, {}):
            raise ValueError(f"Duplicate TP={tp} run at concurrency {concurrency}.")
        grouped[concurrency][tp] = run

    if expected_concurrencies is not None:
        missing_concurrencies = expected_concurrencies - set(grouped)
        if missing_concurrencies:
            raise ValueError(
                "Missing requested concurrency points: "
                + ", ".join(map(str, sorted(missing_concurrencies)))
            )
    if not grouped:
        raise ValueError("No TP=1/TP=2 matrix runs were provided.")

    points: list[MatrixPoint] = []
    matrix_identity: tuple[str, str | None, str | None] | None = None
    for concurrency, by_tp in sorted(grouped.items()):
        if set(by_tp) != {1, 2}:
            raise ValueError(f"Concurrency {concurrency} requires both TP=1 and TP=2 runs.")
        tp1 = by_tp[1]
        tp2 = by_tp[2]
        if (tp1.model.model_id, tp1.model.revision, tp1.model.dtype) != (
            tp2.model.model_id,
            tp2.model.revision,
            tp2.model.dtype,
        ):
            raise ValueError(f"Model identity differs at concurrency {concurrency}.")
        point_identity = (tp1.model.model_id, tp1.model.revision, tp1.model.dtype)
        if matrix_identity is None:
            matrix_identity = point_identity
        elif point_identity != matrix_identity:
            raise ValueError("Matrix contains more than one model identity.")
        if (
            tp1.runtime.enforce_eager,
            tp1.runtime.disable_custom_all_reduce,
            tp1.runtime.max_model_len,
            tp1.runtime.gpu_memory_utilization,
        ) != (
            tp2.runtime.enforce_eager,
            tp2.runtime.disable_custom_all_reduce,
            tp2.runtime.max_model_len,
            tp2.runtime.gpu_memory_utilization,
        ):
            raise ValueError(f"Runtime controls differ at concurrency {concurrency}.")
        tp1_value = metric_value(tp1, metric)
        tp2_value = metric_value(tp2, metric)
        if tp1_value is None or tp2_value is None:
            raise ValueError(f"Metric {metric.value} is missing at concurrency {concurrency}.")
        delta_percent = None if tp1_value == 0 else (tp2_value - tp1_value) / tp1_value * 100
        beneficial = tp2_value < tp1_value if metric in _LOWER_IS_BETTER else tp2_value > tp1_value
        scaling_efficiency = None
        if (
            metric
            in {
                VerificationMetric.OUTPUT_TOKENS_PER_SECOND,
                VerificationMetric.REQUEST_THROUGHPUT_RPS,
            }
            and tp1_value != 0
        ):
            scaling_efficiency = tp2_value / (2.0 * tp1_value)
        points.append(
            MatrixPoint(
                concurrency=concurrency,
                tp1_run_id=tp1.run_id,
                tp2_run_id=tp2.run_id,
                tp1_value=tp1_value,
                tp2_value=tp2_value,
                delta_percent=delta_percent,
                tp_scaling_efficiency=scaling_efficiency,
                tp2_beneficial=beneficial,
            )
        )

    first = next((point for point in points if point.tp2_beneficial), None)
    return CrossoverAnalysis(
        metric=metric,
        crossover_observed=first is not None,
        first_concurrency=first.concurrency if first else None,
        tp1_value=first.tp1_value if first else None,
        tp2_value=first.tp2_value if first else None,
        delta_percent=first.delta_percent if first else None,
        points=points,
        scaling_efficiency_formula=(
            "TP2 throughput / (2 * TP1 throughput); reported only for throughput metrics "
            "and used here as a diagnostic efficiency ratio, not a universal definition."
        ),
    )
