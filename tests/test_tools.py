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
