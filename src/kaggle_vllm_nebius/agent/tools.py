from __future__ import annotations

from dataclasses import dataclass

from kaggle_vllm_nebius.evidence.comparison import compare_runs
from kaggle_vllm_nebius.evidence.schema import EvidenceBundle

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
            run = self.runs[arguments["run_id"]]
            return run.model_dump(mode="json")

        if name == "compare_runs":
            baseline = self.runs[arguments["baseline_run_id"]]
            candidate = self.runs[arguments["candidate_run_id"]]
            return compare_runs(baseline, candidate)

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
