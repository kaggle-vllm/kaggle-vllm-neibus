from fastapi.testclient import TestClient

from kaggle_vllm_nebius.api.app import create_app
from kaggle_vllm_nebius.evidence.loader import load_evidence


def test_api_returns_404_for_unknown_run(tmp_path):
    client = TestClient(create_app(tmp_path / "doctor.db"))
    response = client.get("/api/runs/missing")
    assert response.status_code == 404
    assert response.json() == {"detail": "Run not found: missing"}


def test_api_persists_runs_across_app_instances(tmp_path):
    database = tmp_path / "doctor.db"
    bundle = load_evidence("artifacts/public/normalized/qwen-tp1-c08.json")
    first = TestClient(create_app(database))
    assert first.post("/api/runs", json=bundle.model_dump(mode="json")).status_code == 200

    second = TestClient(create_app(database))
    response = second.get(f"/api/runs/{bundle.run_id}")
    assert response.status_code == 200
    assert response.json()["provenance"]["source_sha256"] == bundle.provenance.source_sha256


def test_api_request_validation_remains_422(tmp_path):
    client = TestClient(create_app(tmp_path / "doctor.db"))
    response = client.post("/api/diagnose", json={"question": "", "run_ids": []})
    assert response.status_code == 422


def test_api_distinguishes_unconfigured_token_factory(tmp_path, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "")
    client = TestClient(create_app(tmp_path / "doctor.db"))
    bundle = load_evidence("artifacts/public/normalized/qwen-tp1-c08.json")
    client.post("/api/runs", json=bundle.model_dump(mode="json"))
    response = client.post(
        "/api/diagnose",
        json={"question": "Diagnose", "run_ids": [bundle.run_id]},
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "Nebius Token Factory is not configured."}
