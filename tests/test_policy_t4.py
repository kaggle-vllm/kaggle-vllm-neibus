from kaggle_vllm_nebius.evidence.loader import load_evidence
from kaggle_vllm_nebius.policy.validator import validate_experiment
from kaggle_vllm_nebius.recommendations.schema import ExperimentSpec


def test_t4_accepts_fp16_tp1():
    evidence = load_evidence("artifacts/demo/tp1-c8.json")
    exp = ExperimentSpec(
        tensor_parallel_size=1,
        concurrency=16,
        dtype="float16",
        rationale="Test",
    )
    result = validate_experiment(exp, evidence)
    assert result.valid


def test_t4_rejects_bfloat16_in_current_policy():
    evidence = load_evidence("artifacts/demo/tp1-c8.json")
    exp = ExperimentSpec(
        tensor_parallel_size=1,
        concurrency=16,
        dtype="bfloat16",
        rationale="Test",
    )
    result = validate_experiment(exp, evidence)
    assert not result.valid
