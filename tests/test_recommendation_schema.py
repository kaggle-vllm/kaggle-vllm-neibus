import pytest
from pydantic import ValidationError

from kaggle_vllm_nebius.recommendations.schema import (
    DiagnosisReport,
    ExperimentSpec,
    validate_diagnosis_for_mode,
)


def test_diagnosis_report_schema():
    report = DiagnosisReport.model_validate(
        {
            "summary": "TP2 was slower in the measured run.",
            "findings": [
                {
                    "claim": "Measured throughput was lower.",
                    "evidence_paths": [
                        "run:tp1.metrics.output_tokens_per_second",
                        "run:tp2.metrics.output_tokens_per_second",
                    ],
                    "confidence": "high",
                }
            ],
            "recommendation": None,
            "warnings": [],
        }
    )
    assert report.findings[0].confidence == "high"


def test_experiment_requires_verification_metrics():
    with pytest.raises(ValidationError):
        ExperimentSpec(
            tensor_parallel_size=1,
            concurrency=8,
            dtype="float16",
            objective="Improve throughput",
            rationale="Controlled experiment",
            verification_metrics=[],
        )


def test_optimization_mode_accepts_specific_missing_evidence_without_recommendation():
    report = DiagnosisReport(
        summary="A controlled recommendation is not yet supported.",
        findings=[],
        insufficient_evidence=["No TP=2 run exists at the baseline concurrency."],
    )
    assert validate_diagnosis_for_mode(report, require_experiment=True) is report


def test_optimization_mode_rejects_empty_null_recommendation():
    report = DiagnosisReport(summary="Unknown", findings=[])
    with pytest.raises(ValueError, match="specific missing evidence"):
        validate_diagnosis_for_mode(report, require_experiment=True)
