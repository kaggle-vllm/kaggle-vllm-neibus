from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AuditTrail:
    events: list[dict[str, Any]] = field(default_factory=list)

    def add(self, event_type: str, **payload: Any) -> None:
        self.events.append({"type": event_type, **payload})

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.events, indent=2), encoding="utf-8")
