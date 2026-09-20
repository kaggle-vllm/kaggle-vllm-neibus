from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SENSITIVE_MARKERS = (
    "api_key",
    "access_token",
    "authorization",
    "credential",
    "password",
    "secret",
)


def _sanitize(value: Any, *, key: str = "") -> Any:
    lowered = key.lower()
    if any(marker in lowered for marker in _SENSITIVE_MARKERS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {item_key: _sanitize(item, key=item_key) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


@dataclass
class AuditTrail:
    events: list[dict[str, Any]] = field(default_factory=list)

    def add(self, event_type: str, **payload: Any) -> None:
        self.events.append(_sanitize({"type": event_type, **payload}))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.events, indent=2), encoding="utf-8")
