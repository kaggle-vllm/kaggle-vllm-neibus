from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


def from_completion(completion) -> Usage:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return Usage(None, None, None)
    return Usage(
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
        getattr(usage, "total_tokens", None),
    )
