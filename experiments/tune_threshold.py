import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
sys.path.append(os.getcwd())

from src.models.CausalTransformer import CausalTransformer
from src.data.SCMGenerator import SCMGenerator
from src.data.CausalDataset import CausalDataset
from src.training.metrics import compute_shd, compute_f1

def tune_threshold(checkpoint_path, num_graphs=8, test_vars=20):
    print(f"--- Tuning Threshold (N={test_vars}) ---")
    
    # Load Model (CPU)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint['args']
    device = torch.device('cpu')
    
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
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Generator
    gen = SCMGenerator(num_nodes=test_vars, edge_prob=0.2, seed=101) # Validation Seed
    
    # Collect All Logits and True Adjacencies
    all_logits = []
    all_true = []
    
    print("Collecting validation data...")
    for i in range(num_graphs):
        pipeline_data = gen.generate_pipeline(
            num_nodes=test_vars, 
            edge_prob=0.2, 
            num_samples_base=32, 
            num_samples_per_intervention=32,
            as_torch=True
        )
        
        true_adj = torch.tensor(nx.to_numpy_array(pipeline_data['dag'])).float()
        
        # Prepare Batch for Model
        base_samples = pipeline_data['base_tensor'][:32]
        int_df = pipeline_data['all_dfs'][1]
        int_mask = pipeline_data['all_masks'][1][0]
        int_samples = torch.tensor(int_df.values, dtype=torch.float32)[:32]
        
        target_row = base_samples
        int_mask_t = torch.tensor(int_mask).float().unsqueeze(0).repeat(32, 1) # Fix for broadcasting
        int_idx = torch.argmax(int_mask_t, dim=1)
        
        with torch.no_grad():
            _, logits, _, _ = model(base_samples, int_samples, target_row, int_mask_t, int_idx)
            avg_logits = logits.mean(dim=0) # (N, N)
            
        all_logits.append(avg_logits)
        all_true.append(true_adj)
        
    all_logits = torch.stack(all_logits)
    all_true = torch.stack(all_true)
    
    # Sweep Thresholds
    thresholds = np.linspace(-5.0, 5.0, 50)
    best_shd = float('inf')
    best_f1 = 0.0
    best_thresh = 0.0
    
    shd_curve = []
    f1_curve = []
    
    print("Sweeping thresholds...")
    for t in thresholds:
        # Vectorized computation
        pred_adj = (all_logits > t).float()
        
        # Compute SHD for batch
        # compute_shd expects (B, N, N)
        shd = compute_shd(pred_adj, all_true) # Returns scalar mean SHD? No, returns sum or mean? 
        # Checking metrics.py: compute_shd returns float (sum over batch usually, or mean?)
        # Let's check implementation. 
        # Usually it iterates.
        # Ideally we want Mean SHD.
        
        # Re-implement simple batch SHD here to be safe/fast
        # diff = |pred - true|. sum dims.
        diff = torch.abs(pred_adj - all_true).sum(dim=(1,2))
        avg_shd = diff.mean().item()
        
        # F1
        tp = (pred_adj * all_true).sum(dim=(1,2))
        fp = (pred_adj * (1-all_true)).sum(dim=(1,2))
        fn = ((1-pred_adj) * all_true).sum(dim=(1,2))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        avg_f1 = f1.mean().item()
        
        shd_curve.append(avg_shd)
        f1_curve.append(avg_f1)
        
        if avg_shd < best_shd:
            best_shd = avg_shd
            best_thresh = t
            best_f1 = avg_f1
            
    print(f"\nOptimization Results:")
    print(f"Best Threshold: {best_thresh:.2f}")
    print(f"Best SHD: {best_shd:.2f}")
    print(f"F1 at Best SHD: {best_f1:.4f}")
    
    # Baseline comparison (Threshold 0.0)
    zero_idx = np.argmin(np.abs(thresholds - 0.0))
    print(f"Default (0.0) SHD: {shd_curve[zero_idx]:.2f}")
    
    return best_thresh

if __name__ == "__main__":
    tune_threshold("final_chekpoint/checkpoint_epoch_253.pt")
