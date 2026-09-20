# Hackathon development statement

The core `kaggle-vllm` compatibility SDK predates the Nebius x NVIDIA Global AI
Hackathon and is maintained separately at:

https://github.com/kaggle-vllm/kaggle-vllm

This submission, `kaggle-vllm-nebius`, is a separate project created during the
hackathon submission period.

The hackathon project adds:

- a Nebius Token Factory control/reasoning plane powered by NVIDIA Nemotron;
- a typed GPU and inference evidence schema;
- read-only agent tools over benchmark/runtime evidence;
- deterministic run comparison;
- typed diagnosis and experiment schemas;
- NVIDIA T4 compatibility policy validation;
- audit trails for model and tool actions;
- an API/CLI for diagnosis and run inspection;
- a closed-loop design for measuring, recommending, rerunning, and verifying
  inference configurations;
- a strict adapter from real `kaggle-vllm-serving-benchmark-v1` artifacts;
- raw-artifact SHA256 provenance and public evidence cataloging;
- deterministic matrix crossover and TP scaling-efficiency analysis;
- objective/constraint-aware `PASS` / `FAIL` / `INCONCLUSIVE` verification;
- SQLite persistence for runs, diagnoses, recommendations, and verification;
- offline CI and expanded adapter/API/policy/agent tests.

The original `kaggle-vllm` repository remains an independent runtime dependency
rather than the hackathon submission itself.
