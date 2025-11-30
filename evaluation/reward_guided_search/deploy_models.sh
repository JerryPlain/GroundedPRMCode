#!/bin/bash

# Script to deploy policy and reward models via vLLM in SLURM environment
# Activate conda environment (adjust path as needed)
source ~/.bashrc
conda activate GroundedPRM

# NOTE: Update these paths to your model directories
POLICY_DIR="${POLICY_DIR:-$HOME/.cache/huggingface/hub}"
REWARD_MODEL_DIR="${REWARD_MODEL_DIR:-/baseline}"

# === User Configurable Section ===
# Set model names (relative to MODEL_BASE_DIR) or None to skip deployment
POLICY_MODEL_NAME="models--Qwen--Qwen2.5-7B-Instruct"
REWARD_MODEL_NAME="Grounded-PRM" 

# Set GPU indices for each model
POLICY_GPU=0
REWARD_GPU=1

# Set ports for each model
POLICY_PORT=8000
REWARD_PORT=8001
# === End User Configurable Section ===

# Create logs directory
mkdir -p logs

# Get SLURM environment variables
NODE_NAME=$(hostname)
echo "Running on node: $NODE_NAME"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"

# Set CUDA device assignment for SLURM
export CUDA_VISIBLE_DEVICES=$POLICY_GPU,$REWARD_GPU

PIDS=()

# Deploy policy model if set
if [ "$POLICY_MODEL_NAME" != "None" ]; then
    POLICY_MODEL_PATH="$POLICY_DIR/$POLICY_MODEL_NAME"
    if [ ! -d "$POLICY_MODEL_PATH" ]; then
        echo "Error: Policy model path does not exist: $POLICY_MODEL_PATH" >&2
        exit 1
    fi
    echo "Deploying policy model: $POLICY_MODEL_NAME on port $POLICY_PORT (GPU $POLICY_GPU)..."
    CUDA_VISIBLE_DEVICES=$POLICY_GPU python -m vllm.entrypoints.openai.api_server \
        --model "$POLICY_MODEL_PATH" \
        --served-model-name "policy-model" \
        --port "$POLICY_PORT" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 32768 \
        --trust-remote-code \
        --api-key "EMPTY" \
        --host 0.0.0.0 &
    POLICY_PID=$!
    PIDS+=("$POLICY_PID")
    echo "Policy model started with PID: $POLICY_PID"
    sleep 15
else
    echo "Policy model deployment skipped."
fi

# Deploy reward model if set
if [ "$REWARD_MODEL_NAME" != "None" ]; then
    REWARD_MODEL_PATH="$REWARD_MODEL_DIR/$REWARD_MODEL_NAME"
    if [ ! -d "$REWARD_MODEL_PATH" ]; then
        echo "Error: Reward model path does not exist: $REWARD_MODEL_PATH" >&2
        exit 1
    fi
    echo "Deploying reward model: $REWARD_MODEL_NAME on port $REWARD_PORT (GPU $REWARD_GPU)..."
    CUDA_VISIBLE_DEVICES=$REWARD_GPU python -m vllm.entrypoints.openai.api_server \
        --model "$REWARD_MODEL_PATH" \
        --served-model-name "reward-model" \
        --port "$REWARD_PORT" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 4096 \
        --trust-remote-code \
        --api-key "EMPTY" \
        --host 0.0.0.0 &
    REWARD_PID=$!
    PIDS+=("$REWARD_PID")
    echo "Reward model started with PID: $REWARD_PID"
    sleep 15
else
    echo "Reward model deployment skipped."
fi

# Check if models are running
if [ "$POLICY_MODEL_NAME" != "None" ]; then
    if pgrep -f "policy-model" > /dev/null; then
        echo "✓ Policy model is running"
        echo "Policy model API: http://$NODE_NAME:$POLICY_PORT/v1"
    else
        echo "✗ Policy model failed to start" >&2
    fi
fi
if [ "$REWARD_MODEL_NAME" != "None" ]; then
    if pgrep -f "reward-model" > /dev/null; then
        echo "✓ Reward model is running"
        echo "Reward model API: http://$NODE_NAME:$REWARD_PORT/v1"
    else
        echo "✗ Reward model failed to start" >&2
    fi
fi

echo ""
echo "Deployment summary:"
echo "Node: $NODE_NAME"
if [ "$POLICY_MODEL_NAME" != "None" ]; then
    echo "Policy model: $POLICY_MODEL_NAME (port $POLICY_PORT, GPU $POLICY_GPU)"
fi
if [ "$REWARD_MODEL_NAME" != "None" ]; then
    echo "Reward model: $REWARD_MODEL_NAME (port $REWARD_PORT, GPU $REWARD_GPU)"
fi

echo "To stop the models, run:"
echo "kill ${PIDS[*]}"
echo "Or cancel the SLURM job: scancel $SLURM_JOB_ID"

echo "Models are running. Press Ctrl+C to stop or wait for SLURM timeout."
wait 