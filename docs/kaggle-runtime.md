# Kaggle runtime

The hackathon application does not replace the validated `kaggle-vllm` runtime.

The existing runtime remains responsible for:

- Kaggle environment validation;
- native CUDA wheel activation;
- upstream vLLM execution;
- NVIDIA T4 / SM75 compatibility;
- NCCL;
- tensor parallelism;
- serving and benchmark evidence.

Nemotron is not hosted in this T4 plane.
