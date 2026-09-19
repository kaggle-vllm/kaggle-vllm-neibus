from kaggle_vllm_nebius.evidence.comparison import compare_runs
from kaggle_vllm_nebius.evidence.loader import load_evidence

baseline = load_evidence("artifacts/demo/tp1-c8.json")
candidate = load_evidence("artifacts/demo/tp2-c8.json")

print(compare_runs(baseline, candidate))
