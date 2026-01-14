#!/bin/bash
# Server Training Script for Generative Causal Foundation Model
# Features enabled:
# - Distributed Data Parallel (DDP)
# - Global MoE Load Balancing
# - Robust RoPE (Variable ID)
# - Domain Randomization (Laplace/Gumbel/Cauchy)
# - I-NLL Optimization

# Ensure virtual environment is activated
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set Master Port to avoid collisions
export MASTER_PORT=${MASTER_PORT:-29505}

# Detect Number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
if [ $? -ne 0 ]; then
    echo "nvidia-smi not found. Defaulting to 1 GPU/CPU."
    NUM_GPUS=1
fi

echo "launching training on $NUM_GPUS GPUs..."

# Launch with torchrun (DDP)
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT main.py \
    --epochs 5000 \
    --batch_size 1 \
    --lr 2e-4 \
    --min_vars 20 \
    --max_vars 100 \
    --num_layers 16 \
    --lambda_aux_moe 0.01 \
    --lambda_delta 100.0 \
    --grad_checkpoint \
    --grad_clip 1.0 \
    --loss_type focal \
    --intervention_prob 0.3 \
    "$@"
