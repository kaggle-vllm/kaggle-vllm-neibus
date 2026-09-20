from __future__ import annotations

from dataclasses import dataclass

from kaggle_vllm_nebius.evidence.comparison import compare_matrix, compare_runs
from kaggle_vllm_nebius.evidence.schema import EvidenceBundle
from kaggle_vllm_nebius.recommendations.schema import VerificationMetric

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_run",
            "description": "Return one measured evidence bundle by run id.",
            "parameters": {
                "type": "object",
                "properties": {"run_id": {"type": "string"}},
                "required": ["run_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_gpu_telemetry",
            "description": "Return normalized per-GPU telemetry for one measured run.",
            "parameters": {
                "type": "object",
                "properties": {"run_id": {"type": "string"}},
                "required": ["run_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_request_distribution",
            "description": "Return measured request/token/latency distributions for one run.",
            "parameters": {
                "type": "object",
                "properties": {"run_id": {"type": "string"}},
                "required": ["run_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_runtime_identity",
            "description": "Return source, runtime, model, and topology provenance for one run.",
            "parameters": {
                "type": "object",
                "properties": {"run_id": {"type": "string"}},
                "required": ["run_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_crossover_analysis",
            "description": "Deterministically find the first TP=2 benefit in the loaded matrix.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": [item.value for item in VerificationMetric],
                    },
                    "concurrencies": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                    },
                },
                "required": ["metric"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_runs",
            "description": "Compare two measured benchmark runs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "baseline_run_id": {"type": "string"},
                    "candidate_run_id": {"type": "string"},
                },
                "required": ["baseline_run_id", "candidate_run_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_compatibility_contract",
            "description": "Return validated Kaggle Tesla T4 runtime constraints.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


@dataclass
class ToolRegistry:
    runs: dict[str, EvidenceBundle]

    def execute(self, name: str, arguments: dict) -> dict:
        if name == "get_run":
            run = self._run(arguments["run_id"])
            return run.model_dump(mode="json")

        if name == "compare_runs":
            baseline = self._run(arguments["baseline_run_id"])
            candidate = self._run(arguments["candidate_run_id"])
            return compare_runs(baseline, candidate)

        if name == "get_gpu_telemetry":
            run = self._run(arguments["run_id"])
            return run.gpu_metrics.model_dump(mode="json")

        if name == "get_request_distribution":
            run = self._run(arguments["run_id"])
            return {
                "workload": run.workload.model_dump(mode="json"),
                "latency_metrics_ms": {
                    "ttft_p50_ms": run.metrics.ttft_p50_ms,
                    "ttft_p95_ms": run.metrics.ttft_p95_ms,
                    "tpot_p50_ms": run.metrics.tpot_p50_ms,
                    "tpot_p95_ms": run.metrics.tpot_p95_ms,
                },
                "distribution": (
                    run.request_distribution.model_dump(mode="json")
                    if run.request_distribution
                    else None
                ),
            }

        if name == "get_runtime_identity":
            run = self._run(arguments["run_id"])
            return {
                "provenance": run.provenance.model_dump(mode="json"),
                "environment": run.environment.model_dump(mode="json"),
                "gpus": [gpu.model_dump(mode="json") for gpu in run.gpus],
                "model": run.model.model_dump(mode="json"),
                "runtime": run.runtime.model_dump(mode="json"),
            }

        if name == "get_crossover_analysis":
            analysis = compare_matrix(
                self.runs.values(),
                metric=VerificationMetric(arguments["metric"]),
                concurrencies=arguments.get("concurrencies"),
            )
            return analysis.model_dump(mode="json")

        if name == "get_compatibility_contract":
            return {
                "validated_platform": "Kaggle Notebook",
                "gpus": "2 x NVIDIA Tesla T4",
                "compute_capability": "7.5 / SM75",
                "validated_dtype": "float16",
                "validated_tensor_parallel_sizes": [1, 2],
                "attention_backend_observed": "TRITON_ATTN",
                "flash_attention_2": False,
                "important_non_claims": [
                    "No H100/NVFP4 compatibility is claimed for Kaggle T4.",
                    "TP=2 is not universally faster than TP=1.",
                    (
                        "The benchmark evidence may show correlation without "
                        "isolating a single root cause."
                    ),
                ],
            }

        raise KeyError(f"Unknown tool: {name}")

    def _run(self, run_id: str) -> EvidenceBundle:
        try:
            return self.runs[run_id]
        except KeyError as exc:
            raise KeyError(f"Unknown run id: {run_id}") from exc
