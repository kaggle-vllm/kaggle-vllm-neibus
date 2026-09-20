# Agent loop

1. User asks an inference-engineering question.
2. Nemotron sees available run IDs.
3. Nemotron requests read-only tools.
4. Local Python executes tools against validated evidence.
5. Tool results are returned to Nemotron.
6. Nemotron emits one `DiagnosisReport` JSON object.
7. Pydantic validates it.
8. The deterministic policy validates any proposed experiment.
9. A later controlled run is imported as measured evidence.
10. Python evaluates every declared verification metric and constraint.
11. A typed `VerificationReport` records `PASS`, `FAIL`, or `INCONCLUSIVE`.

Optimization mode requires either a non-empty, policy-checkable
`ExperimentSpec` or a specific `insufficient_evidence` explanation. It does not
force a fabricated recommendation.
