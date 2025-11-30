#!/bin/bash

# Script to deploy LLM API service via vLLM
# Activate conda environment
source ~/.bashrc
conda activate grounded_prm

# NOTE: Update MODEL_PATH to your model path
MODEL_PATH="${MODEL_PATH:-/path/to/your/model}"

# Create logs directory
mkdir -p logs

# Start API service
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name policy_model \
  --trust-remote-code \
  --port 8007 \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768