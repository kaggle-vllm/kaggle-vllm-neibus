from kaggle_vllm_nebius.recommendations.schema import DiagnosisReport


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
