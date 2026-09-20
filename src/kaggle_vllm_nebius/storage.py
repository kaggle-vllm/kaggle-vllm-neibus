from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

from kaggle_vllm_nebius.evidence.schema import EvidenceBundle
from kaggle_vllm_nebius.recommendations.schema import DiagnosisReport
from kaggle_vllm_nebius.recommendations.verify import VerificationReport


class SQLiteStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    evidence_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS diagnosis_sessions (
                    session_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    run_ids_json TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    audit_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS recommendations (
                    session_id TEXT PRIMARY KEY,
                    experiment_json TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES diagnosis_sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS verification_results (
                    verification_id TEXT PRIMARY KEY,
                    baseline_run_id TEXT NOT NULL,
                    experiment_run_id TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def put_run(self, bundle: EvidenceBundle) -> None:
        payload = bundle.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO runs(run_id, evidence_json) VALUES (?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET evidence_json=excluded.evidence_json",
                (bundle.run_id, payload),
            )

    def get_run(self, run_id: str) -> EvidenceBundle | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT evidence_json FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return EvidenceBundle.model_validate_json(row["evidence_json"]) if row else None

    def list_run_ids(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT run_id FROM runs ORDER BY run_id").fetchall()
        return [row["run_id"] for row in rows]

    def save_diagnosis(
        self,
        *,
        question: str,
        run_ids: list[str],
        report: DiagnosisReport,
        audit: list[dict],
    ) -> str:
        session_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO diagnosis_sessions"
                "(session_id, question, run_ids_json, report_json, audit_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    question,
                    json.dumps(run_ids),
                    report.model_dump_json(),
                    json.dumps(audit),
                ),
            )
            if report.recommendation is not None:
                connection.execute(
                    "INSERT INTO recommendations(session_id, experiment_json) VALUES (?, ?)",
                    (session_id, report.recommendation.model_dump_json()),
                )
        return session_id

    def save_verification(self, report: VerificationReport) -> str:
        verification_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO verification_results"
                "(verification_id, baseline_run_id, experiment_run_id, report_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    verification_id,
                    report.baseline_run_id,
                    report.experiment_run_id,
                    report.model_dump_json(),
                ),
            )
        return verification_id
