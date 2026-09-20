import json
from pathlib import Path
from types import SimpleNamespace

from kaggle_vllm_nebius.agent.audit import AuditTrail
from kaggle_vllm_nebius.agent.orchestrator import InferenceDoctor
from kaggle_vllm_nebius.agent.tools import ToolRegistry
from kaggle_vllm_nebius.config import Settings
from kaggle_vllm_nebius.evidence.loader import load_evidence


class FakeClient:
    def __init__(self, _settings):
        self.calls = 0

    def chat(self, *, messages, tools):
        self.calls += 1
        if self.calls == 1:
            function = SimpleNamespace(
                name="get_crossover_analysis",
                arguments=json.dumps(
                    {
                        "metric": "output_tokens_per_second",
                        "concurrencies": [1, 8, 16, 32],
                    }
                ),
            )
            message = SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(id="call-1", function=function)],
            )
        else:
            report = {
                "summary": "TP=2 first exceeded TP=1 at concurrency 16.",
                "findings": [
                    {
                        "claim": "The deterministic crossover was observed at concurrency 16.",
                        "evidence_paths": ["tool:get_crossover_analysis.first_concurrency"],
                        "confidence": "high",
                    }
                ],
                "inferences": [],
                "insufficient_evidence": [],
                "recommendation": {
                    "tensor_parallel_size": 2,
                    "concurrency": 16,
                    "dtype": "float16",
                    "objective": "Confirm the observed throughput crossover.",
                    "rationale": "Repeat the controlled measured configuration.",
                    "verification_metrics": ["output_tokens_per_second"],
                    "constraints": [],
                },
                "warnings": [],
            }
            message = SimpleNamespace(content=json.dumps(report), tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class CorrectingClient:
    def __init__(self, _settings):
        self.calls = 0

    def chat(self, *, messages, tools):
        self.calls += 1
        recommendation = {
            "tensor_parallel_size": 2,
            "concurrency": 16,
            "dtype": "float16",
            "objective": "Improve throughput.",
            "rationale": "Controlled retry.",
            "verification_metrics": (
                [{"metric": "output_tokens_per_second"}]
                if self.calls == 1
                else ["output_tokens_per_second"]
            ),
            "constraints": [],
        }
        report = {
            "summary": "Measured crossover.",
            "findings": [],
            "inferences": [],
            "insufficient_evidence": [],
            "recommendation": recommendation,
            "warnings": [],
        }
        if self.calls == 1:
            report["optimization_mode"] = True
        message = SimpleNamespace(content=json.dumps(report), tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_mocked_nemotron_tool_loop(monkeypatch):
    monkeypatch.setattr(
        "kaggle_vllm_nebius.agent.orchestrator.NebiusClient",
        FakeClient,
    )
    runs = {}
    for path in Path("artifacts/public/normalized").glob("*.json"):
        run = load_evidence(path)
        runs[run.run_id] = run
    audit = AuditTrail()
    doctor = InferenceDoctor(
        Settings("fake", "https://example.invalid/v1", "model", 4),
        ToolRegistry(runs),
        audit=audit,
    )
    report = doctor.diagnose("Optimize TP crossover", require_experiment=True)
    assert report.recommendation is not None
    assert report.recommendation.verification_metrics == ["output_tokens_per_second"]
    assert any(event["type"] == "tool_call" for event in audit.events)


def test_agent_requests_schema_correction_after_invalid_final_json(monkeypatch):
    monkeypatch.setattr(
        "kaggle_vllm_nebius.agent.orchestrator.NebiusClient",
        CorrectingClient,
    )
    run = load_evidence("artifacts/public/normalized/qwen-tp1-c16.json")
    audit = AuditTrail()
    doctor = InferenceDoctor(
        Settings("fake", "https://example.invalid/v1", "model", 3),
        ToolRegistry({run.run_id: run}),
        audit=audit,
    )
    report = doctor.diagnose("Optimize", require_experiment=True)
    assert report.recommendation is not None
    assert any(event["type"] == "model_validation_failed" for event in audit.events)
