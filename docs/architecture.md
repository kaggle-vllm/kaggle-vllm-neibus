# Architecture

The project deliberately separates two inference planes.

## Execution plane

`kaggle-vllm` runs upstream vLLM on the validated Kaggle dual-T4 profile and
produces benchmark, runtime, NCCL, and GPU evidence.

## Reasoning plane

NVIDIA Nemotron 3.5 Lightning runs through Nebius Token Factory. It receives
only structured evidence through read-only tools.

## Control boundary

Nemotron returns a typed `DiagnosisReport` and optional `ExperimentSpec`.
A deterministic policy validator decides whether the experiment fits the
validated Kaggle T4 compatibility contract.

The model is never granted unrestricted shell access.
