from __future__ import annotations

import argparse
import json
import platform
import sys

from kaggle_vllm_nebius.agent.audit import AuditTrail
from kaggle_vllm_nebius.agent.orchestrator import InferenceDoctor
from kaggle_vllm_nebius.agent.tools import ToolRegistry
from kaggle_vllm_nebius.config import load_settings
from kaggle_vllm_nebius.evidence.comparison import compare_runs
from kaggle_vllm_nebius.evidence.loader import load_evidence
from kaggle_vllm_nebius.nebius.client import NebiusClient
from kaggle_vllm_nebius.policy.validator import validate_experiment


def _print_json(value) -> None:
    print(json.dumps(value, indent=2, default=str))


def cmd_doctor(_args) -> int:
    settings = load_settings()
    print("kaggle-vllm-nebius local doctor")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Token Factory configured: {bool(settings.nebius_api_key)}")
    print(f"Token Factory base URL: {settings.nebius_base_url}")
    print(f"Nemotron model: {settings.nebius_model}")

    if sys.version_info[:2] != (3, 11):
        print(
            "WARNING: this local setup is intended to be developed with "
            "your Python 3.11 installation."
        )
    return 0


def cmd_validate(args) -> int:
    bundle = load_evidence(args.path)
    _print_json(bundle.model_dump(mode="json"))
    return 0


def cmd_compare(args) -> int:
    baseline = load_evidence(args.baseline)
    candidate = load_evidence(args.candidate)
    _print_json(compare_runs(baseline, candidate))
    return 0


def cmd_smoke(_args) -> int:
    settings = load_settings()
    client = NebiusClient(settings)
    print(client.smoke())
    return 0


def cmd_diagnose(args) -> int:
    baseline = load_evidence(args.baseline)
    candidate = load_evidence(args.candidate)
    runs = {baseline.run_id: baseline, candidate.run_id: candidate}

    audit = AuditTrail()
    doctor = InferenceDoctor(
        load_settings(),
        ToolRegistry(runs),
        audit=audit,
    )
    report = doctor.diagnose(args.question)
    _print_json(report.model_dump(mode="json"))

    if report.recommendation is not None:
        policy = validate_experiment(report.recommendation, baseline)
        print("\nPolicy validation:")
        _print_json(policy.__dict__)

    if args.audit:
        audit.save(args.audit)
        print(f"\nAudit trail: {args.audit}")

    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run(
        "kaggle_vllm_nebius.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle-vllm-nebius",
        description="Evidence-driven NVIDIA GPU inference diagnostics.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("validate")
    p.add_argument("path")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("compare")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("token-factory-smoke")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("diagnose")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--question", required=True)
    p.add_argument(
        "--audit",
        default="artifacts/private/latest-audit.json",
    )
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
