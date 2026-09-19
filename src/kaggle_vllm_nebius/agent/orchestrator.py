from __future__ import annotations

import json
import re
from typing import Any

from kaggle_vllm_nebius.agent.audit import AuditTrail
from kaggle_vllm_nebius.agent.system_prompt import SYSTEM_PROMPT
from kaggle_vllm_nebius.agent.tools import TOOL_SCHEMAS, ToolRegistry
from kaggle_vllm_nebius.config import Settings
from kaggle_vllm_nebius.nebius.client import NebiusClient
from kaggle_vllm_nebius.recommendations.schema import DiagnosisReport


def _json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Nemotron response did not contain a JSON object")
        return json.loads(match.group(0))


class InferenceDoctor:
    def __init__(
        self,
        settings: Settings,
        tools: ToolRegistry,
        audit: AuditTrail | None = None,
    ):
        self.settings = settings
        self.client = NebiusClient(settings)
        self.tools = tools
        self.audit = audit or AuditTrail()

    def diagnose(self, question: str) -> DiagnosisReport:
        run_ids = sorted(self.tools.runs)
        schema = DiagnosisReport.model_json_schema()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Available run ids: {run_ids}\n\n"
                    "Investigate with tools. Your final response must be exactly "
                    "one JSON object validating against this JSON Schema:\n"
                    f"{json.dumps(schema)}"
                ),
            },
        ]

        self.audit.add("agent_started", question=question, run_ids=run_ids)

        for turn in range(self.settings.max_agent_turns):
            completion = self.client.chat(messages=messages, tools=TOOL_SCHEMAS)
            message = completion.choices[0].message
            self.audit.add(
                "model_turn",
                turn=turn,
                content=message.content,
                tool_call_count=len(message.tool_calls or []),
            )

            # Append an OpenAI-compatible assistant message.
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": message.content,
            }
            if message.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                ]
            messages.append(assistant_message)

            if not message.tool_calls:
                if not message.content:
                    raise RuntimeError("Nemotron returned no final content.")
                report = DiagnosisReport.model_validate(_json_from_text(message.content))
                self.audit.add(
                    "agent_finished",
                    report=report.model_dump(mode="json"),
                )
                return report

            for call in message.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                result = self.tools.execute(call.function.name, args)
                self.audit.add(
                    "tool_call",
                    name=call.function.name,
                    arguments=args,
                    result=result,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result),
                    }
                )

        raise RuntimeError(
            f"Agent exceeded {self.settings.max_agent_turns} turns "
            "without producing a final DiagnosisReport."
        )
