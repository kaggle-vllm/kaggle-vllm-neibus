from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from kaggle_vllm_nebius.agent.audit import AuditTrail
from kaggle_vllm_nebius.agent.orchestrator import InferenceDoctor
from kaggle_vllm_nebius.agent.tools import ToolRegistry
from kaggle_vllm_nebius.config import load_settings
from kaggle_vllm_nebius.evidence.comparison import compare_runs
from kaggle_vllm_nebius.evidence.schema import EvidenceBundle
from kaggle_vllm_nebius.policy.validator import validate_experiment
from kaggle_vllm_nebius.recommendations.schema import ExperimentSpec
from kaggle_vllm_nebius.recommendations.verify import verify_experiment
from kaggle_vllm_nebius.storage import SQLiteStore


class DiagnoseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    run_ids: list[str] = Field(min_length=1)
    require_experiment: bool = False


class VerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_run_id: str
    experiment_run_id: str
    specification: ExperimentSpec


def create_app(database_path: str | Path | None = None) -> FastAPI:
    db_path = database_path or os.getenv(
        "KAGGLE_VLLM_NEBIUS_DB",
        "artifacts/private/kaggle-vllm-nebius.db",
    )
    store = SQLiteStore(db_path)
    application = FastAPI(
        title="Kaggle-vLLM Inference Doctor",
        version="0.1.0",
    )
    application.state.store = store

    def require_run(run_id: str) -> EvidenceBundle:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return run

    @application.get("/api/health")
    def health() -> dict:
        settings = load_settings()
        return {
            "ok": True,
            "token_factory_configured": bool(settings.nebius_api_key),
            "model": settings.nebius_model,
            "stored_runs": len(store.list_run_ids()),
        }

    @application.post("/api/runs")
    def ingest_run(run: EvidenceBundle) -> dict:
        store.put_run(run)
        return {"stored": run.run_id}

    @application.get("/api/runs")
    def list_runs() -> dict:
        return {"runs": store.list_run_ids()}

    @application.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> EvidenceBundle:
        return require_run(run_id)

    @application.get("/api/compare/{baseline_id}/{candidate_id}")
    def compare(baseline_id: str, candidate_id: str) -> dict:
        return compare_runs(require_run(baseline_id), require_run(candidate_id))

    @application.post("/api/diagnose")
    def diagnose(request: DiagnoseRequest) -> dict:
        selected = {run_id: require_run(run_id) for run_id in request.run_ids}
        audit = AuditTrail()
        try:
            doctor = InferenceDoctor(
                load_settings(),
                ToolRegistry(selected),
                audit=audit,
            )
            report = doctor.diagnose(
                request.question,
                require_experiment=request.require_experiment,
            )
        except RuntimeError as exc:
            message = str(exc)
            if "NEBIUS_API_KEY is not set" in message:
                raise HTTPException(
                    status_code=503,
                    detail="Nebius Token Factory is not configured.",
                ) from exc
            raise HTTPException(status_code=502, detail="Diagnosis failed.") from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(
                status_code=502, detail="Diagnosis returned invalid evidence."
            ) from exc

        session_id = store.save_diagnosis(
            question=request.question,
            run_ids=request.run_ids,
            report=report,
            audit=audit.events,
        )
        policy = None
        if report.recommendation is not None:
            policy = validate_experiment(
                report.recommendation,
                next(iter(selected.values())),
            ).__dict__
        return {
            "session_id": session_id,
            "report": report.model_dump(mode="json"),
            "policy": policy,
        }

    @application.post("/api/verify")
    def verify(request: VerifyRequest) -> dict:
        report = verify_experiment(
            require_run(request.baseline_run_id),
            require_run(request.experiment_run_id),
            request.specification,
        )
        verification_id = store.save_verification(report)
        return {
            "verification_id": verification_id,
            "report": report.model_dump(mode="json"),
        }

    return application


app = create_app()
