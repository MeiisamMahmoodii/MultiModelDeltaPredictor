# ISD-CP: Interventional Structure Discovery & Causal Prediction

**Unified Causal Learning via Physics-First Transformer**

## Overview
ISD-CP is a novel architecture designed to solve the **Delta Prediction Problem** in causal inference:
$$f(X, I) \to \Delta$$
Unlike traditional methods that prioritize graph discovery ($A$), ISD-CP prioritizes predicting the *consequences* of interventions. By solving the physics directly, the causal structure emerges implicitly within the model's attention mechanism.

## Key Features
*   **Physics-First Strategy**: We treat causal inference as a regression problem first, structure learning second.
*   **Decoupled Training**: Phase 1 trains the physics engine ($f$) without DAG constraints. Phase 2 refines the structure.
*   **Causal Transformer**: A customized Transformer with Rotary Positional Embeddings (RoPE) and Learned Causal Masking.
*   **Mixture of Experts (MoE)**: A Hard-Gumbel MoE layer to capture diverse physical mechanisms (linear, polynomial, step functions) simultaneously.
*   **Adaptive Interventions**: Interventions are scaled dynamically by variable variance ($\sigma$) to ensure robust generalization.

## Project Structure
*   `src/`: Core source code.
    *   `src/models/CausalTransformer.py`: The main model architecture.
    *   `src/training/`: Loss functions, curriculum manager, and metrics.
    *   `src/data/`: SCM generation and dataset handling.
*   `docs/`: Documentation and Reports.
    *   `docs/supervisor_report.md`: High-level architectural summary and status.
    *   `docs/ARCHIVE_LOGS.md`: Technical history of debugging and stabilization.
    *   `docs/RESEARCH_NOTES.md`: Bibliography and literature review.
*   `experiments/`: Benchmark suites and evaluation scripts.

## Quick Start
To train the model (Phase 1: Physics Only):

```bash
# Clean start
pkill -f main.py
rm -rf checkpoints/checkpoint_epoch_*.pt

CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 --standalone main.py \
  --epochs 300 \
  --batch_size 64 \
  --lr 2e-5 \
  --min_vars 20 --max_vars 50 \
  --intervention_prob 0.5 \
  --lambda_dag 0.0 --lambda_h 0.0 --lambda_sparse 0.0 \
  --lambda_aux_moe 0.2 \
  --router_tau 1.2
```

## Current Status
*   **Phase**: Phase 1 (Physics Execution)
*   **Performance**: MAE ~5.2 (Level 20 / 40 Variables).
*   **Stability**: Robust against Gradient Explosion (via Clipping) and Expert Collapse (via Load Balancing).

## License
MIT License
