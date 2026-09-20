from pathlib import Path

from kaggle_vllm_nebius.evidence.comparison import compare_matrix
from kaggle_vllm_nebius.evidence.loader import load_evidence


def real_matrix():
    return [
        load_evidence(path) for path in sorted(Path("artifacts/public/normalized").glob("*.json"))
    ]


def test_real_qwen_matrix_detects_measured_crossover():
    result = compare_matrix(real_matrix(), concurrencies=[1, 8, 16, 32])
    assert result.crossover_observed
    assert result.first_concurrency == 16
    assert result.tp1_value == 138.74944716864707
    assert result.tp2_value == 174.27246539500263
    assert result.points[-1].tp_scaling_efficiency == 0.8439474098445018


def test_matrix_reports_no_crossover():
    runs = real_matrix()
    adjusted = []
    for run in runs:
        if run.runtime.tensor_parallel_size == 2:
            tp1 = next(
                candidate
                for candidate in runs
                if candidate.runtime.tensor_parallel_size == 1
                and candidate.workload.concurrency == run.workload.concurrency
            )
            run = run.model_copy(
                update={
                    "metrics": run.metrics.model_copy(
                        update={
                            "output_tokens_per_second": (tp1.metrics.output_tokens_per_second * 0.5)
                        }
                    )
                }
            )
        adjusted.append(run)

    result = compare_matrix(adjusted, concurrencies=[1, 8, 16, 32])
    assert not result.crossover_observed
    assert result.first_concurrency is None
