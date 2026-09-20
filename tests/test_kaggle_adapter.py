import hashlib
import json
from pathlib import Path

import pytest

from kaggle_vllm_nebius.adapters.kaggle_vllm_serving import (
    SOURCE_SCHEMA,
    KaggleServingArtifactError,
    import_kaggle_serving_artifact,
)
from kaggle_vllm_nebius.cli import build_parser
from kaggle_vllm_nebius.evidence.loader import load_evidence

RAW = Path("artifacts/public/raw/qwen-tp1-c08.json")


def test_real_serving_artifact_import_and_provenance(tmp_path):
    output = tmp_path / "normalized" / "run.json"
    bundle = import_kaggle_serving_artifact(RAW, output)
    expected_sha = hashlib.sha256(RAW.read_bytes()).hexdigest()

    assert bundle.provenance.source_schema == SOURCE_SCHEMA
    assert bundle.provenance.source_sha256 == expected_sha
    assert bundle.runtime.tensor_parallel_size == 1
    assert bundle.workload.successful_requests == 24
    assert bundle.workload.failed_requests == 0
    assert bundle.metrics.ttft_p95_ms == pytest.approx(11695.949558999928)
    assert bundle.gpu_metrics.peak_memory_mib == 13731.0
    assert (tmp_path / "raw" / RAW.name).read_bytes() == RAW.read_bytes()
    assert load_evidence(output).run_id == bundle.run_id


def test_adapter_rejects_invalid_source_schema(tmp_path):
    payload = json.loads(RAW.read_text())
    payload["schema_version"] = "unknown"
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps(payload))

    with pytest.raises(KaggleServingArtifactError, match="Unsupported source schema"):
        import_kaggle_serving_artifact(source, tmp_path / "out.json")


def test_adapter_rejects_missing_required_metrics(tmp_path):
    payload = json.loads(RAW.read_text())
    payload["measurements"]["output_throughput_tokens_per_second"] = None
    source = tmp_path / "missing.json"
    source.write_text(json.dumps(payload))

    with pytest.raises(KaggleServingArtifactError, match="Required measured fields"):
        import_kaggle_serving_artifact(source, tmp_path / "out.json")


def test_import_cli_writes_normalized_and_preserved_raw(tmp_path):
    output = tmp_path / "normalized" / "run.json"
    args = build_parser().parse_args(
        [
            "import-kaggle-run",
            "--input",
            str(RAW),
            "--output",
            str(output),
        ]
    )
    assert args.func(args) == 0
    assert output.exists()
    assert (tmp_path / "raw" / RAW.name).exists()
