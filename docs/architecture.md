# Architecture

The project deliberately separates two inference planes.

## Execution plane

`kaggle-vllm` runs upstream vLLM on the validated Kaggle dual-T4 profile and
produces benchmark, runtime, NCCL, topology, and GPU evidence. The strict adapter
preserves raw bytes and emits typed evidence with source SHA256 provenance.

## Reasoning plane

NVIDIA Nemotron 3.5 Lightning runs through Nebius Token Factory. It receives
only bounded structured evidence through read-only tools. Python computes
deltas, crossover points, scaling efficiency, policy decisions, and verification;
Nemotron interprets those facts.

## Control boundary

Nemotron returns a typed `DiagnosisReport` and optional `ExperimentSpec`.
A deterministic policy validator decides whether the experiment fits the
validated Kaggle T4 compatibility contract.

The model is never granted unrestricted shell access. SQLite persists evidence,
diagnosis sessions, recommendations, and verification reports. A future Nebius
GPU backend is optional and does not block the present evidence workflow.
