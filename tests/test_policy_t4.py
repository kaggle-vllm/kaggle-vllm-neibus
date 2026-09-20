from kaggle_vllm_nebius.evidence.loader import load_evidence
from kaggle_vllm_nebius.policy.commands import render_benchmark_command
from kaggle_vllm_nebius.policy.validator import validate_experiment
from kaggle_vllm_nebius.recommendations.schema import ExperimentSpec


def experiment(**overrides):
    values = {
        "tensor_parallel_size": 1,
        "concurrency": 16,
        "dtype": "float16",
        "objective": "Improve measured throughput.",
        "rationale": "Controlled test",
        "verification_metrics": ["output_tokens_per_second"],
    }
    values.update(overrides)
    return ExperimentSpec(**values)


def test_t4_accepts_fp16_tp1():
    evidence = load_evidence("artifacts/demo/tp1-c8.json")
    exp = experiment()
    result = validate_experiment(exp, evidence)
    assert result.valid


def test_t4_rejects_bfloat16_in_current_policy():
    evidence = load_evidence("artifacts/demo/tp1-c8.json")
    exp = experiment(dtype="bfloat16")
    result = validate_experiment(exp, evidence)
    assert not result.valid


def test_t4_rejects_tp_above_available_gpu_count():
    evidence = load_evidence("artifacts/demo/tp1-c8.json")
    result = validate_experiment(experiment(tensor_parallel_size=3), evidence)
    assert not result.valid
    assert any("available GPU count" in error for error in result.errors)


def test_high_memory_utilization_warns():
    evidence = load_evidence("artifacts/demo/tp1-c8.json")
    result = validate_experiment(experiment(gpu_memory_utilization=0.96), evidence)
    assert result.valid
    assert any("headroom" in warning for warning in result.warnings)


def test_invalid_experiment_cannot_render_command():
    evidence = load_evidence("artifacts/demo/tp1-c8.json")
    try:
        render_benchmark_command("model", "out.json", experiment(dtype="bfloat16"), evidence)
    except ValueError as exc:
        assert "rejected by policy" in str(exc)
    else:
        raise AssertionError("invalid experiment unexpectedly rendered")
