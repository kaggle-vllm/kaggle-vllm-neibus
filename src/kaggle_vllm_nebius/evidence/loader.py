from __future__ import annotations

import json
from pathlib import Path

from .schema import EvidenceBundle


def load_evidence(path: str | Path) -> EvidenceBundle:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    return EvidenceBundle.model_validate(payload)


def save_evidence(bundle: EvidenceBundle, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
