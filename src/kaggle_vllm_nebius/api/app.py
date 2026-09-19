from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from kaggle_vllm_nebius.agent.audit import AuditTrail
from kaggle_vllm_nebius.agent.orchestrator import InferenceDoctor
from kaggle_vllm_nebius.agent.tools import ToolRegistry
from kaggle_vllm_nebius.config import load_settings
from kaggle_vllm_nebius.evidence.comparison import compare_runs
from kaggle_vllm_nebius.evidence.schema import EvidenceBundle

app = FastAPI(
    title="Kaggle-vLLM Inference Doctor",
    version="0.1.0",
)

RUNS: dict[str, EvidenceBundle] = {}


class DiagnoseRequest(BaseModel):
    question: str
    run_ids: list[str]


@app.get("/api/health")
def health() -> dict:
    settings = load_settings()
    return {
        "ok": True,
        "token_factory_configured": bool(settings.nebius_api_key),
        "model": settings.nebius_model,
        "stored_runs": len(RUNS),
    }


@app.post("/api/runs")
def ingest_run(run: EvidenceBundle) -> dict:
    RUNS[run.run_id] = run
    return {"stored": run.run_id}


@app.get("/api/runs")
def list_runs() -> dict:
    return {"runs": sorted(RUNS)}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> EvidenceBundle:
    return RUNS[run_id]


@app.get("/api/compare/{baseline_id}/{candidate_id}")
def compare(baseline_id: str, candidate_id: str) -> dict:
    return compare_runs(RUNS[baseline_id], RUNS[candidate_id])


@app.post("/api/diagnose")
def diagnose(request: DiagnoseRequest) -> dict:
    selected = {run_id: RUNS[run_id] for run_id in request.run_ids}
    audit = AuditTrail()
    doctor = InferenceDoctor(
        load_settings(),
        ToolRegistry(selected),
        audit=audit,
    )
    report = doctor.diagnose(request.question)
    return {
        "report": report.model_dump(mode="json"),
        "audit": audit.events,
    }
