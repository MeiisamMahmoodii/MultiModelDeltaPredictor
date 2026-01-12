import torch
import numpy as np
from src.data.SCMGenerator import SCMGenerator

def debug_tensors():
    # Use same params as failing Phase 2
    # lambda_dag=100.0, lambda_sparse=0.5
    generator = SCMGenerator(num_nodes=50, edge_prob=0.2, noise_scale=1.0)
    
    print("Generating data batch (N=50)...")
    data = generator.generate_pipeline(
        num_samples_base=100, 
        num_samples_per_intervention=100,
        intervention_prob=0.5
    )
    
    # Extract Base and Deltas
    base_df = data['all_dfs'][0]
    base_vals = base_df.values
    
    all_deltas = []
    all_int_vals = []
    
    for i, int_df in enumerate(data['all_dfs'][1:]):
        int_vals = int_df.values
        delta = int_vals - base_vals
        all_deltas.append(delta)
        all_int_vals.append(int_vals)
        
    base_flat = base_vals.flatten()
    deltas_flat = np.concatenate(all_deltas).flatten()
    int_flat = np.concatenate(all_int_vals).flatten()
    
    print("\n--- Tensor Statistics ---")
    print(f"Base Data: Mean={np.mean(base_flat):.4f}, Std={np.std(base_flat):.4f}, Min={np.min(base_flat):.4f}, Max={np.max(base_flat):.4f}")
    print(f"Int Data:  Mean={np.mean(int_flat):.4f}, Std={np.std(int_flat):.4f}, Min={np.min(int_flat):.4f}, Max={np.max(int_flat):.4f}")
    print(f"Deltas:    Mean={np.mean(deltas_flat):.4f}, Std={np.std(deltas_flat):.4f}, Min={np.min(deltas_flat):.4f}, Max={np.max(deltas_flat):.4f}")
    print(f"MAE(Zero): {np.mean(np.abs(deltas_flat)):.4f}")
    
    # Check for "Zero Delta" dominance (Sparsity of effect)
    zero_deltas = np.sum(np.abs(deltas_flat) < 1e-4)
    total_params = len(deltas_flat)
    print(f"Sparsity of Effect: {zero_deltas}/{total_params} ({zero_deltas/total_params*100:.2f}%) are exactly zero.")


if __name__ == "__main__":
    debug_tensors()
