# Evidence schema

`EvidenceBundle` is the project boundary between raw GPU execution and model
reasoning.

Important rules:

1. Missing values stay `null`; do not estimate them.
2. Preserve the raw benchmark artifact separately.
3. Record environment and runtime provenance.
4. Do not turn causal hypotheses into measured fields.
5. Checksum submission-quality raw artifacts.
