from kaggle_vllm_nebius.agent.tools import ToolRegistry
from kaggle_vllm_nebius.evidence.loader import load_evidence


def test_read_only_tool_registry():
    tp1 = load_evidence("artifacts/demo/tp1-c8.json")
    tp2 = load_evidence("artifacts/demo/tp2-c8.json")
    tools = ToolRegistry({tp1.run_id: tp1, tp2.run_id: tp2})

    result = tools.execute(
        "compare_runs",
        {
            "baseline_run_id": tp1.run_id,
            "candidate_run_id": tp2.run_id,
        },
    )
    assert result["same_model"] is True


def test_crossover_tool_uses_deterministic_matrix():
    runs = {}
    for tp in (1, 2):
        for concurrency in (1, 8, 16, 32):
            run = load_evidence(f"artifacts/public/normalized/qwen-tp{tp}-c{concurrency:02d}.json")
            runs[run.run_id] = run
    tools = ToolRegistry(runs)
    result = tools.execute(
        "get_crossover_analysis",
        {
            "metric": "output_tokens_per_second",
            "concurrencies": [1, 8, 16, 32],
        },
    )
    assert result["first_concurrency"] == 16


def test_runtime_identity_tool_exposes_provenance_not_raw_logs():
    run = load_evidence("artifacts/public/normalized/qwen-tp1-c08.json")
    result = ToolRegistry({run.run_id: run}).execute(
        "get_runtime_identity",
        {"run_id": run.run_id},
    )
    assert result["provenance"]["source_schema"] == "kaggle-vllm-serving-benchmark-v1"
    assert "server_log" not in result


def test_request_distribution_tool_includes_measured_latency_breakdown():
    run = load_evidence("artifacts/public/normalized/qwen-tp1-c08.json")
    result = ToolRegistry({run.run_id: run}).execute(
        "get_request_distribution",
        {"run_id": run.run_id},
    )
    assert result["latency_metrics_ms"]["ttft_p95_ms"] == run.metrics.ttft_p95_ms
    assert result["latency_metrics_ms"]["tpot_p95_ms"] == run.metrics.tpot_p95_ms
