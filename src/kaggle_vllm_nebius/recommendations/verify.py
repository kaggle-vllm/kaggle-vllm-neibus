from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from kaggle_vllm_nebius.evidence.schema import EvidenceBundle
from kaggle_vllm_nebius.recommendations.schema import (
    ExperimentSpec,
    MetricConstraint,
    VerificationMetric,
)


class VerificationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class MetricVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: VerificationMetric
    baseline: float | None
    experiment: float | None
    delta: float | None
    delta_percent: float | None
    expected_direction: str
    passed: bool | None
    reason: str


class VerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: VerificationStatus
    baseline_run_id: str
    experiment_run_id: str
    objective: str
    precondition_results: list[str]
    metrics: list[MetricVerification]
    constraint_results: list[str]
    reasons: list[str]


_LOWER_IS_BETTER = {
    VerificationMetric.TTFT_P50_MS,
    VerificationMetric.TTFT_P95_MS,
    VerificationMetric.TPOT_P50_MS,
    VerificationMetric.TPOT_P95_MS,
    VerificationMetric.PEAK_GPU_MEMORY_MIB,
}


def _metric_value(bundle: EvidenceBundle, metric: VerificationMetric) -> float | None:
    if metric == VerificationMetric.AVG_GPU_UTILIZATION_PCT:
        return bundle.gpu_metrics.avg_utilization_pct
    if metric == VerificationMetric.PEAK_GPU_MEMORY_MIB:
        return bundle.gpu_metrics.peak_memory_mib
    return getattr(bundle.metrics, metric.value)


def _check_constraint(value: float, constraint: MetricConstraint) -> bool:
    operations = {
        "lt": value < constraint.value,
        "le": value <= constraint.value,
        "gt": value > constraint.value,
        "ge": value >= constraint.value,
    }
    return operations[constraint.operator]


def verify_experiment(
    baseline: EvidenceBundle,
    experiment: EvidenceBundle,
    specification: ExperimentSpec,
) -> VerificationReport:
    precondition_results: list[str] = []
    preconditions_valid = True

    def check(condition: bool, message: str) -> None:
        nonlocal preconditions_valid
        preconditions_valid = preconditions_valid and condition
        precondition_results.append(f"{'PASS' if condition else 'INCONCLUSIVE'}: {message}")

    check(
        baseline.model.model_id == experiment.model.model_id
        and baseline.model.revision == experiment.model.revision,
        "baseline and experiment model identity/revision match",
    )
    check(
        experiment.runtime.tensor_parallel_size == specification.tensor_parallel_size,
        "experiment TP matches ExperimentSpec",
    )
    check(
        experiment.workload.concurrency == specification.concurrency,
        "experiment concurrency matches ExperimentSpec",
    )
    dtype_aliases = {
        "half": "float16",
        "fp16": "float16",
    }
    measured_dtype = dtype_aliases.get(
        (experiment.model.dtype or "").lower(),
        (experiment.model.dtype or "").lower(),
    )
    specified_dtype = dtype_aliases.get(
        specification.dtype.lower(),
        specification.dtype.lower(),
    )
    check(measured_dtype == specified_dtype, "experiment dtype matches ExperimentSpec")

    metric_results: list[MetricVerification] = []
    reasons: list[str] = []

    for metric in specification.verification_metrics:
        baseline_value = _metric_value(baseline, metric)
        experiment_value = _metric_value(experiment, metric)
        direction = "lower" if metric in _LOWER_IS_BETTER else "higher"
        if baseline_value is None or experiment_value is None:
            reason = f"{metric.value} is missing from one or both measured runs."
            reasons.append(reason)
            metric_results.append(
                MetricVerification(
                    metric=metric,
                    baseline=baseline_value,
                    experiment=experiment_value,
                    delta=None,
                    delta_percent=None,
                    expected_direction=direction,
                    passed=None,
                    reason=reason,
                )
            )
            continue

        delta = experiment_value - baseline_value
        delta_percent = None if baseline_value == 0 else delta / baseline_value * 100.0
        passed = delta < 0 if metric in _LOWER_IS_BETTER else delta > 0
        reason = (
            f"{metric.value} moved in the expected {direction}-is-better direction."
            if passed
            else f"{metric.value} did not improve in the expected direction."
        )
        metric_results.append(
            MetricVerification(
                metric=metric,
                baseline=baseline_value,
                experiment=experiment_value,
                delta=delta,
                delta_percent=delta_percent,
                expected_direction=direction,
                passed=passed,
                reason=reason,
            )
        )

    constraint_results: list[str] = []
    constraint_failures = False
    constraint_missing = False
    for constraint in specification.constraints:
        value = _metric_value(experiment, constraint.metric)
        if value is None:
            constraint_missing = True
            message = f"Constraint metric {constraint.metric.value} is missing."
            reasons.append(message)
            constraint_results.append(message)
            continue
        passed = _check_constraint(value, constraint)
        constraint_failures = constraint_failures or not passed
        constraint_results.append(
            f"{constraint.metric.value}={value} {constraint.operator} "
            f"{constraint.value}: {'PASS' if passed else 'FAIL'}"
        )

    metric_failures = any(item.passed is False for item in metric_results)
    metric_missing = any(item.passed is None for item in metric_results)
    if not preconditions_valid:
        status = VerificationStatus.INCONCLUSIVE
        reasons.append("The measured experiment does not match the declared ExperimentSpec.")
    elif metric_failures or constraint_failures:
        status = VerificationStatus.FAIL
    elif metric_missing or constraint_missing:
        status = VerificationStatus.INCONCLUSIVE
    else:
        status = VerificationStatus.PASS

    if not reasons:
        reasons.append("All declared metrics improved and all deterministic constraints passed.")

    return VerificationReport(
        status=status,
        baseline_run_id=baseline.run_id,
        experiment_run_id=experiment.run_id,
        objective=specification.objective,
        precondition_results=precondition_results,
        metrics=metric_results,
        constraint_results=constraint_results,
        reasons=reasons,
    )
