import torch
import numpy as np
import pandas as pd
import time
import os
import sys
import matplotlib.pyplot as plt
import networkx as nx
sys.path.append(os.getcwd())

from src.models.CausalTransformer import CausalTransformer
from src.data.SCMGenerator import SCMGenerator
from src.training.metrics import compute_shd, compute_f1

def run_budgeted_decoding(checkpoint_path, N=20):
    device = torch.device('cpu') 
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ckpt['args']
    
    # Load Model
    model = CausalTransformer(
        num_nodes=args.max_vars + 5,
        d_model=512,
        num_layers=args.num_layers,
        grad_checkpoint=getattr(args, 'grad_checkpoint', False),
        ablation_dense=getattr(args, 'ablation_dense_moe', False),
        ablation_no_interleaved=getattr(args, 'ablation_no_interleaved', False),
        ablation_no_dag=getattr(args, 'ablation_no_dag', False),
        ablation_no_physics=getattr(args, 'ablation_no_physics', False)
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"--- Budgeted Decoding Experiment (N={N}) ---")
    gen = SCMGenerator(num_nodes=N, edge_prob=0.2, seed=42)
    
    num_graphs = 20
    results = []
    
    # Budgets to test: k = factor * N (e.g., 1N = 20 edges, 2N=40 edges)
    # Expected density p=0.2 => 0.2 * N^2 = 0.2 * 400 = 80 edges.
    # Ah, density is relative to N^2.
    # Degree is 0.2 * N = 4.
    # So expected edges E = 4 * N.
    # We test factors of Expected Edges.
    expected_edges = int(0.2 * N * N)
    budget_factors = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    budgets = [int(f * expected_edges) for f in budget_factors]
    
    print(f"Testing Budgets (Expected {expected_edges}): {budgets}")
    
    for _ in range(num_graphs):
        # Generate Data
        pipeline_data = gen.generate_pipeline(N, 0.2, num_samples_base=200, num_samples_per_intervention=200, as_torch=True)
        true_dag = nx.to_numpy_array(pipeline_data['dag'])
        true_t = torch.tensor(true_dag).float()
        
        num_samples = pipeline_data['base_tensor'].shape[0]
        slice_size = min(32, num_samples)
        
        base_samples = pipeline_data['base_tensor'][:slice_size]
        int_df = pipeline_data['all_dfs'][1]
        int_mask = pipeline_data['all_masks'][1][0]
        int_samples = torch.tensor(int_df.values, dtype=torch.float32)[:slice_size]
        
        target_row = base_samples
        int_mask_t = torch.tensor(int_mask).float().unsqueeze(0).repeat(slice_size, 1) # (B, N)
        int_idx = torch.argmax(int_mask_t, dim=1)[:slice_size]
        
        with torch.no_grad():
            _, logits, _, _, _ = model(base_samples, int_samples, target_row, int_mask_t)
            avg_logits = logits.mean(dim=0) # (N, N)
            
        # 1. Baseline: Fixed Threshold (1.33)
        adj_base = (avg_logits > 1.33).float()
        shd_base = compute_shd(adj_base.unsqueeze(0), true_t.unsqueeze(0))
        results.append({"Method": "Threshold 1.33", "Value": 1.33, "SHD": shd_base})
        
        # 2. Budgeted Decoding (Top-K)
        flat_logits = avg_logits.flatten()
        
        for k in budgets:
            if k == 0: continue
            # Top K
            topk_vals, topk_ind = torch.topk(flat_logits, k)
            # Create adj
            adj_k = torch.zeros_like(flat_logits)
            adj_k[topk_ind] = 1.0
            adj_k = adj_k.view(N, N)
            
            shd = compute_shd(adj_k.unsqueeze(0), true_t.unsqueeze(0))
            results.append({"Method": "Budgeted", "Value": k, "SHD": shd})
            
    df = pd.DataFrame(results)
    print("\nResults Summary:")
    print(df.groupby(["Method", "Value"]).mean())
    df.to_csv("experiment_budgeted_decoding.csv", index=False)

if __name__ == "__main__":
    run_budgeted_decoding("final_chekpoint/checkpoint_epoch_253.pt", N=20)
