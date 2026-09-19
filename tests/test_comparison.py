from kaggle_vllm_nebius.evidence.comparison import compare_runs
from kaggle_vllm_nebius.evidence.loader import load_evidence


def test_comparison_detects_lower_candidate_throughput():
    tp1 = load_evidence("artifacts/demo/tp1-c8.json")
    tp2 = load_evidence("artifacts/demo/tp2-c8.json")
    result = compare_runs(tp1, tp2)
    assert result["output_tokens_per_second"]["percent"] == -25.0
