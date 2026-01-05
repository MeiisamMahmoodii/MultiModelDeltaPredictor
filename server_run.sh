#!/bin/bash
# Server Training Script for ISD-CP
# Usage: bash server_run.sh

# 1. Update Code
echo "--- Pulling Latest Code ---"
git pull origin main

# 2. Setup Venv (if needed)
if [ ! -d ".venv" ]; then
    echo "--- Creating Virtual Env ---"
    python3 -m venv .venv
fi
source .venv/bin/activate

# 3. Install Dependencies
echo "--- Installing Dependencies ---"
pip install -r requirements.txt

# 4. Run Sparsity Finetuning (High-SHD Fix)
echo "--- Starting Sparsity Finetuning ---"
# Note: Ensure checkpoint_epoch_253.pt is on the server at final_chekpoint/
if [ ! -f "final_chekpoint/checkpoint_epoch_253.pt" ]; then
    echo "WARNING: Checkpoint file not found on server!"
    echo "Please SCP the checkpoint: scp final_chekpoint/checkpoint_epoch_253.pt user@server:path/final_chekpoint/"
    exit 1
fi

python main.py \
  --resume \
  --checkpoint_path "checkpoints/checkpoint_epoch_253.pt" \
  --lr 1e-5 \
  --lambda_dag 1000.0 \
  --lambda_sparse 0.1 \
  --epochs 1000 \
  --grad_checkpoint \
  --batch_size 32

echo "--- Training Complete ---"
