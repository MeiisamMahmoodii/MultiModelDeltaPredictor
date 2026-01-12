import torch
import numpy as np
from src.data.SCMGenerator import SCMGenerator

def check_dumb_baselines():
    generator = SCMGenerator(num_nodes=50, edge_prob=0.2)
    
    print("Generating data for N=50...")
    data = generator.generate_pipeline(
        num_samples_base=1024, 
        num_samples_per_intervention=1024,
        intervention_prob=0.5
    )
    
    # Extract Deltas
    # all_dfs is [base, int1, int2...]
    base_df = data['all_dfs'][0]
    base_tensor = torch.tensor(base_df.values, dtype=torch.float32)
    
    all_deltas = []
    
    # Skip index 0 (Base)
    for i, int_df in enumerate(data['all_dfs'][1:]):
        int_tensor = torch.tensor(int_df.values, dtype=torch.float32)
        true_delta = int_tensor - base_tensor
        all_deltas.append(true_delta)
        
    all_deltas = torch.cat(all_deltas, dim=0) # (Total_Samples, 50)
    
    # 1. Predict Zero
    mae_zero = torch.mean(torch.abs(all_deltas - 0.0)).item()
    
    # 2. Predict Mean Delta (Global)
    mean_delta = torch.mean(all_deltas)
    mae_mean = torch.mean(torch.abs(all_deltas - mean_delta)).item()
    
    # 3. Predict Random (Gaussian matching variance)
    std_delta = torch.std(all_deltas)
    random_preds = torch.randn_like(all_deltas) * std_delta + mean_delta
    mae_random = torch.mean(torch.abs(all_deltas - random_preds)).item()
    
    print(f"\n--- Trivial Baselines (N=50) ---")
    print(f"True Delta Range: [{all_deltas.min():.2f}, {all_deltas.max():.2f}]")
    print(f"Predict Zero MAE: {mae_zero:.4f}")
    print(f"Predict Mean MAE: {mae_mean:.4f}")
    print(f"Predict Random MAE: {mae_random:.4f}")
    print(f"--------------------------------")

if __name__ == "__main__":
    check_dumb_baselines()
