import torch
import argparse
import os
import sys
sys.path.append(os.getcwd()) # Ensure src is importable
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader

from src.models.CausalTransformer import CausalTransformer
from src.data.SCMGenerator import SCMGenerator
from src.data.CausalDataset import CausalDataset
from src.data.collate import collate_fn_pad
from src.training.metrics import compute_shd, compute_f1, compute_mae, compute_sid

class RandomBaseline:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
    
    def predict(self, batch_size, device):
        # Random Logits (Normal dist centered at -2 for sparse)
        logits = torch.randn(batch_size, self.num_nodes, self.num_nodes, device=device) - 2.0
        # Zero Deltas
        deltas = torch.zeros(batch_size, self.num_nodes, device=device)
        return deltas, logits

class LinearBaseline:
    """
    Fits a Ridge Regression model on the Context (Base Samples) to predict Deltas?
    No, in our setting, we predict Delta(Int) given Base and Int request.
    Linear Model: Delta = W * Is_Intervened? No.
    
    Physics Baseline: 
    Model: Delta_j = Sum_i (W_ij * (X_i_int - X_i_base)) ? No.
    
    Let's try a simple "Linear SCM" approximation.
    Input: Concatenation of Base Sample and Intervention Mask.
    Output: Delta.
    
    We fit this on a small "calibration" set from the same distribution, 
    or we train it online?
    For a fair comparison with a PRE-TRAINED model, we should train this Linear Model 
    on the *same amount* of data or assumed "converged".
    
    For simplicity: We'll train a fresh Ridge regressor on 1000 samples for EACH graph 
    (Oracle access to graph's data distribution) and test on 100.
    This gives the baseline a slight advantage (per-graph training) vs Generalist Transformer.
    """
    def __init__(self):
        self.model = Ridge(alpha=1.0)
        self.fitted = False
        
    def fit_and_predict(self, gen_func, num_samples_train, num_samples_test, device):
        # Generate Training Data for this specific graph
        # This baseline is "Instance-Optimized" (Strong baseline)
        pass

def evaluate_model(checkpoint_path, batch_size=32, num_graphs=8, samples_per_graph=32, device_name="cpu"):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device_name, weights_only=False)
    args = checkpoint['args']
    
    device = torch.device(device_name)
    
    # Initialize Model
    # Note: Checkpoint args might not have ablation flags if they are new.
    # We use getattr with defaults.
    model = CausalTransformer(
        num_nodes=args.max_vars + 5,
        d_model=512, # Hardcoded in main.py, assuming same
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
    
    # Generate Test Set (Fixed Seed)
    # Using args.max_vars to test on the hardest difficulty seen training?
    # Or args.min_vars?
    # Let's test on max_vars (50).
    test_vars = args.max_vars
    print(f"Evaluating on {num_graphs} graphs with {test_vars} variables...")
    
    # Optimized for Quick Evaluation:
    # Reduce intervention probability to ensure we skip quickly to next graph
    # 50 nodes * 0.05 = 2.5 interventions per graph.
    # 2.5 ints * 8 samples = 20 samples per graph.
    # 8 graphs * 20 = 160 samples = 5 batches.
    
    gen = SCMGenerator(num_nodes=test_vars, edge_prob=0.2, seed=12345) 
    dataset = CausalDataset(
        gen, num_nodes_range=(test_vars, test_vars),
        samples_per_graph=8, # Reduced from 32
        infinite=False, validation_graphs=num_graphs,
        intervention_prob=0.05 # Override args.intervention_prob for speed
    )
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_pad)
    
    results = []
    
    total_metrics = {
        model: {"mae": [], "shd": [], "f1": [], "sid": []},
        "random": {"mae": [], "shd": [], "f1": [], "sid": []}
    }
    
    rand_baseline = RandomBaseline(test_vars)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            # Prepare Inputs
            base = batch['base_samples'].to(device)
            int_s = batch['int_samples'].to(device)
            target = batch['target_row'].to(device)
            mask = batch['int_mask'].to(device)
            idx = batch['int_node_idx'].to(device)
            
            true_delta = batch['delta'].to(device)
            true_adj = batch['adj'].to(device)
            
            # --- 1. Model ---
            # Unpack 5 values
            deltas, logits, _, _, _ = model(base, int_s, target, mask)
            
            # Metrics
            mae = compute_mae(deltas, true_delta)
            shd = compute_shd(logits, true_adj)
            f1 = compute_f1(logits, true_adj)
            # SID is expensive, maybe run on subset?
            # Running on full batch for now.
            sid = compute_sid(logits, true_adj)
            
            total_metrics[model]["mae"].append(mae)
            total_metrics[model]["shd"].append(shd)
            total_metrics[model]["f1"].append(f1)
            total_metrics[model]["sid"].append(sid)
            
            # --- 2. Random Baseline ---
            deltas_rand, logits_rand = rand_baseline.predict(base.shape[0], device)
            
            mae_r = compute_mae(deltas_rand, true_delta)
            shd_r = compute_shd(logits_rand, true_adj)
            f1_r = compute_f1(logits_rand, true_adj)
            sid_r = compute_sid(logits_rand, true_adj)
            
            total_metrics["random"]["mae"].append(mae_r)
            total_metrics["random"]["shd"].append(shd_r)
            total_metrics["random"]["f1"].append(f1_r)
            total_metrics["random"]["sid"].append(sid_r)
            
    # Compile Report
    summary = {}
    for name, metrics in [("Model (ISD-CP)", total_metrics[model]), ("Random Baseline", total_metrics["random"])]:
        summary[name] = {
            "MAE": np.mean(metrics["mae"]),
            "SHD": np.mean(metrics["shd"]),
            "SID": np.mean(metrics["sid"]),
            "F1": np.mean(metrics["f1"])
        }
        
    df = pd.DataFrame(summary).T
    print("\n--- Benchmark Results ---")
    print(df)
    
    # Save to Markdown
    md = f"# Benchmark Report: Epoch {checkpoint['epoch']}\n\n"
    md += "## Configuration\n"
    md += f"- Variables: {test_vars}\n"
    md += f"- Graphs: {num_graphs}\n"
    md += f"- Samples/Graph: {samples_per_graph}\n\n"
    md += "## Results\n"
    md += df.to_markdown()
    
    with open("benchmark_report.md", "w") as f:
        f.write(md)
    print("Report saved to benchmark_report.md")

if __name__ == "__main__":
    evaluate_model("final_chekpoint/checkpoint_epoch_253.pt", device_name="mps" if torch.backends.mps.is_available() else "cpu")
