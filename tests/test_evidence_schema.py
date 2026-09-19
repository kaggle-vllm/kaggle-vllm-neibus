import json
from pathlib import Path

from kaggle_vllm_nebius.evidence.schema import EvidenceBundle


def test_demo_evidence_validates():
    payload = json.loads(Path("artifacts/demo/tp1-c8.json").read_text())
    bundle = EvidenceBundle.model_validate(payload)
    assert bundle.runtime.tensor_parallel_size == 1
    assert len(bundle.gpus) == 2
