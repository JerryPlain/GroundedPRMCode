#!/bin/bash
#SBATCH --job-name=deepseek
#SBATCH --output=logs/deepseek_output_%j.log
#SBATCH --error=logs/deepseek_error_%j.log
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --gres=gpu:2
#SBATCH --qos=mcml
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00         


# Activate conda environment
# NOTE: Update CONDA_ENV_PATH to your conda environment path
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$HOME/conda-envs/llama_factory}"
source ~/.bashrc
conda activate "$CONDA_ENV_PATH"

# Create logs directory
mkdir -p logs

# Start API service (blocking)
# NOTE: Update MODEL_PATH to your model path
MODEL_PATH="${MODEL_PATH:-/path/to/your/model}"
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name qwen_distill_32b \
  --trust-remote-code \
  --port 8003 \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768