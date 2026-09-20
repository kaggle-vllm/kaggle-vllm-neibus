#!/usr/bin/env python3
"""Development-only manual EvidenceBundle builder.

Production imports must use ``kaggle-vllm-nebius import-kaggle-run`` so the raw
source is validated, preserved, and checksummed. This compatibility script is
kept only for older examples that do not have a supported raw source artifact.
"""

from __future__ import annotations

import argparse

from kaggle_vllm_nebius.evidence.loader import save_evidence
from kaggle_vllm_nebius.evidence.schema import (
    EnvironmentInfo,
    EvidenceBundle,
    GPUInfo,
    GPUMetrics,
    ModelInfo,
    PerformanceMetrics,
    Provenance,
    RuntimeInfo,
    WorkloadInfo,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, choices=[1, 2], required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--output-tps", type=float)
    p.add_argument("--ttft-p95-ms", type=float)
    p.add_argument("--tpot-p95-ms", type=float)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    bundle = EvidenceBundle(
        run_id=args.run_id,
        provenance=Provenance(
            project="kaggle-vllm",
            project_version="0.2.0",
        ),
        environment=EnvironmentInfo(
            platform="Kaggle Notebook",
            python_version="3.12.13",
            pytorch_version="2.10.0+cu128",
            cuda_version="12.8",
            vllm_version="0.18.1 source baseline",
        ),
        gpus=[
            GPUInfo(
                index=0,
                name="NVIDIA Tesla T4",
                compute_capability="7.5",
                memory_mib=15360,
            ),
            GPUInfo(
                index=1,
                name="NVIDIA Tesla T4",
                compute_capability="7.5",
                memory_mib=15360,
            ),
        ],
        model=ModelInfo(
            model_id=args.model,
            dtype="float16",
        ),
        runtime=RuntimeInfo(
            tensor_parallel_size=args.tp,
            attention_backend="TRITON_ATTN",
            enforce_eager=True,
            disable_custom_all_reduce=True,
        ),
        workload=WorkloadInfo(
            concurrency=args.concurrency,
        ),
        metrics=PerformanceMetrics(
            output_tokens_per_second=args.output_tps,
            ttft_p95_ms=args.ttft_p95_ms,
            tpot_p95_ms=args.tpot_p95_ms,
        ),
        gpu_metrics=GPUMetrics(),
    )
    save_evidence(bundle, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
