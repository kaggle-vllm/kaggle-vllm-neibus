from kaggle_vllm_nebius.evidence.loader import load_evidence
from kaggle_vllm_nebius.recommendations.schema import ExperimentSpec
from kaggle_vllm_nebius.recommendations.verify import (
    VerificationStatus,
    verify_experiment,
)


def specification(*metrics: str) -> ExperimentSpec:
    return ExperimentSpec(
        tensor_parallel_size=2,
        concurrency=16,
        dtype="float16",
        objective="Improve measured serving throughput.",
        rationale="Validate TP=2 at the observed crossover.",
        verification_metrics=list(metrics),
        constraints=[{"metric": "peak_gpu_memory_mib", "operator": "le", "value": 14848}],
    )


def test_verification_pass():
    baseline = load_evidence("artifacts/public/normalized/qwen-tp1-c16.json")
    experiment = load_evidence("artifacts/public/normalized/qwen-tp2-c16.json")
    report = verify_experiment(
        baseline,
        experiment,
        specification("output_tokens_per_second", "request_throughput_rps"),
    )
    assert report.status == VerificationStatus.PASS


def test_verification_fail():
    baseline = load_evidence("artifacts/public/normalized/qwen-tp1-c16.json")
    experiment = load_evidence("artifacts/public/normalized/qwen-tp2-c16.json")
    experiment = experiment.model_copy(
        update={
            "metrics": experiment.metrics.model_copy(update={"output_tokens_per_second": 100.0})
        }
    )
    report = verify_experiment(
        baseline,
        experiment,
        specification("output_tokens_per_second"),
    )
    assert report.status == VerificationStatus.FAIL


def test_verification_inconclusive_when_declared_metric_missing():
    baseline = load_evidence("artifacts/demo/tp1-c8.json")
    experiment = load_evidence("artifacts/demo/tp2-c8.json")
    report = verify_experiment(
        baseline,
        experiment,
        specification("peak_gpu_memory_mib"),
    )
    assert report.status == VerificationStatus.INCONCLUSIVE
    assert report.metrics[0].passed is None


def test_verification_inconclusive_when_run_does_not_match_specification():
    baseline = load_evidence("artifacts/public/normalized/qwen-tp1-c08.json")
    experiment = load_evidence("artifacts/public/normalized/qwen-tp2-c08.json")
    report = verify_experiment(
        baseline,
        experiment,
        specification("output_tokens_per_second"),
    )
    assert report.status == VerificationStatus.INCONCLUSIVE
    assert any("concurrency" in result for result in report.precondition_results)
