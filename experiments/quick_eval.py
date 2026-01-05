import torch
import argparse
import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models.CausalTransformer import CausalTransformer
from src.data.SCMGenerator import SCMGenerator
from src.data.CausalDataset import CausalDataset
from src.data.collate import collate_fn_pad
from src.training.metrics import compute_shd, compute_f1, compute_mae, compute_sid

def evaluate_model(checkpoint_path, device_name="cpu"):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device_name, weights_only=False)
    args = checkpoint['args']
    
    device = torch.device(device_name)
    
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
    
    # Ultra-Fast Eval Settings
    num_graphs = 2
    samples_per_graph = 16 
    test_vars = args.max_vars
    
    print(f"Evaluating on {num_graphs} graphs with {test_vars} variables...")
    
    # Low intervention prob to ensure sparse targets
    gen = SCMGenerator(num_nodes=test_vars, edge_prob=0.2, seed=12345) 
    dataset = CausalDataset(
        gen, num_nodes_range=(test_vars, test_vars),
        samples_per_graph=samples_per_graph,
        infinite=False, validation_graphs=num_graphs,
        intervention_prob=0.05 
    )
    loader = CausalDataset( # Using dataset directly to control loop manually?
       # No, rely on DataLoader
       gen, num_nodes_range=(test_vars, test_vars),
       samples_per_graph=samples_per_graph,
       infinite=False, validation_graphs=num_graphs,
       intervention_prob=0.05
    )
    # Use DataLoader for collation
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn_pad)

    total_metrics = {
        "Model": {"mae": [], "shd": [], "f1": []},
        "Random": {"mae": [], "shd": [], "f1": []}
    }
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            # Force Limit
            if i >= 4: break 
            
            base = batch['base_samples'].to(device)
            int_s = batch['int_samples'].to(device)
            target = batch['target_row'].to(device)
            mask = batch['int_mask'].to(device)
            idx = batch['int_node_idx'].to(device)
            true_delta = batch['delta'].to(device)
            true_adj = batch['adj'].to(device)
            
            # Model Predict
            deltas, logits, _, _ = model(base, int_s, target, mask, idx)
            
            mae = compute_mae(deltas, true_delta)
            shd = compute_shd(logits, true_adj)
            f1 = compute_f1(logits, true_adj)
            
            total_metrics["Model"]["mae"].append(mae)
            total_metrics["Model"]["shd"].append(shd)
            total_metrics["Model"]["f1"].append(f1)
            
            # Random Baseline
            logits_rand = torch.randn_like(logits) - 2.0
            deltas_rand = torch.zeros_like(deltas)
            
            mae_r = compute_mae(deltas_rand, true_delta)
            shd_r = compute_shd(logits_rand, true_adj)
            f1_r = compute_f1(logits_rand, true_adj)
            
            total_metrics["Random"]["mae"].append(mae_r)
            total_metrics["Random"]["shd"].append(shd_r)
            total_metrics["Random"]["f1"].append(f1_r)

    # Compile Report
    summary = {}
    for name, metrics in total_metrics.items():
        summary[name] = {
            "MAE": np.mean(metrics["mae"]),
            "SHD": np.mean(metrics["shd"]),
            "F1": np.mean(metrics["f1"])
        }
        
    df = pd.DataFrame(summary).T
    print("\n--- Benchmark Results ---")
    print(df)
    
    # Save to Markdown
    md = f"# Benchmark Report: Epoch {checkpoint['epoch']}\n\n"
    md += "## Configuration\n"
    md += f"- Variables: {test_vars}\n"
    md += f"- Evaluation Set: {num_graphs} Graphs\n\n"
    md += "## Results\n"
    md += df.to_markdown()
    
    with open("benchmark_report.md", "w") as f:
        f.write(md)

if __name__ == "__main__":
    evaluate_model("final_chekpoint/checkpoint_epoch_253.pt", device_name="mps" if torch.backends.mps.is_available() else "cpu")
