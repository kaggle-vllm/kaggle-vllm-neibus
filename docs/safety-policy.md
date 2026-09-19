# Safety and execution policy

The agent is intentionally evidence-first and read-only.

It cannot execute arbitrary model-generated shell commands. Experiment settings
must be represented as `ExperimentSpec` and pass deterministic validation.

For the current T4 profile:

- validated TP sizes: 1 and 2;
- validated dtype: FP16;
- observed attention backend: Triton;
- H100-specific NVFP4 / FlashInfer / DSpark reference settings are not treated
  as compatible with the Kaggle T4 runtime.
