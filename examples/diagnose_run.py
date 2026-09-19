from kaggle_vllm_nebius.agent.audit import AuditTrail
from kaggle_vllm_nebius.agent.orchestrator import InferenceDoctor
from kaggle_vllm_nebius.agent.tools import ToolRegistry
from kaggle_vllm_nebius.config import load_settings
from kaggle_vllm_nebius.evidence.loader import load_evidence

tp1 = load_evidence("artifacts/demo/tp1-c8.json")
tp2 = load_evidence("artifacts/demo/tp2-c8.json")

audit = AuditTrail()
doctor = InferenceDoctor(
    load_settings(),
    ToolRegistry({tp1.run_id: tp1, tp2.run_id: tp2}),
    audit=audit,
)

report = doctor.diagnose("Why did TP=2 underperform TP=1?")
print(report.model_dump_json(indent=2))
audit.save("artifacts/private/example-audit.json")
