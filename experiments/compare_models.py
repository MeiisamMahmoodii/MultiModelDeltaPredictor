import torch
import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

# Baselines
from src.models.baselines.notears import NotearsLinear
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz

# Our Model
from src.models.CausalTransformer import CausalTransformer
from src.data.SCMGenerator import SCMGenerator
from src.data.CausalDataset import CausalDataset
from src.data.collate import collate_fn_pad
from src.training.metrics import compute_shd, compute_f1

def run_pc(data_np):
    # PC Algorithm
    # data_np: (n_samples, n_vars)
    # Returns: adj matrix (numpy)
    cg = pc(data_np, 0.05, fisherz, True, 0, -1) # True = stable
    # Convert Graph to Adjacency
    # cg.G.graph is numpy array?
    # cg.G.graph[i, j] = -1/1 (Tail/Arrow)
    # 1 -> -1 (Arrow at i from j?) No.
    # causal-learn G format:
    # 1: Circle, 2: Arrow, 3: Tail
    # -1: Null
    # Standard output: graph
    adj = cg.G.graph 
    # Convert to standard binary adjacency where A[j, i] = 1 means j->i
    # Causal Learn: graph[i, j] is endpoint at j.
    # Endpoint types: 0 (Null), 1 (Circle), 2 (Arrow), 3 (Tail)
    # A[i, j] = 1 if j->i. NO.
    # Convention: A[i, j] = 1 if i -> j.
    
    n = data_np.shape[1]
    res_adj = np.zeros((n, n))
    
    # Iterate over all pairs
    for i in range(n):
        for j in range(n):
            # Edge i - j
            # check endpoint at j (from i)
            end_j = adj[i, j] 
            # check endpoint at i (from j)
            end_i = adj[j, i]
            
            # Directed Edge i -> j means:
            # i has tail (3) or circle? and j has arrow (2)
            if end_j == 2 and end_i == 3:
                res_adj[i, j] = 1
            # Keep undirected/bidirected as zero for SHD? 
            # Or assume CP-DAG results.
            # Measuring SHD against DAG requires treating undirected as error?
            # Standard SHD counts undirected edges.
            
    return res_adj

def run_notears(data_torch, device):
    # NOTEARS
    # data_torch: (n_samples, n_vars)
    nt = NotearsLinear(d=data_torch.shape[1], max_iter=20) # Low iter for speed in quick eval
    # Need to move logic to CPU/GPU? Notears implementation above is naive pytorch.
    # Assuming CPU for LBFGS stability usually.
    W = nt.fit(data_torch.cpu())
    # W_ij means i -> j
    # Thresholding done in fit.
    return (W != 0).astype(float)

def evaluate_baselines(checkpoint_path, num_graphs=3, test_vars=20):
    print(f"--- Comparing Models (N={test_vars}) ---")
    
    # Load Ours
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint['args']
    # Force CPU for baselines evaluation to avoid MPS weirdness with causallearn/numpy interop
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
    model.to(device) # CPU
    model.eval()
    
    # Generator
    gen = SCMGenerator(num_nodes=test_vars, edge_prob=0.2, seed=42)
    
    # We need access to the FULL observational data for PC/Notears
    # CausalDataset yields small batches of Interventional Data.
    # Baselines (Standard) usually work on pure Observational Data.
    # But ISD-CP uses Interventional Data.
    # To be fair, we should give Baselines the Observational part of the data?
    # PC/Notears are designed for Obs data (mostly).
    # "Soft" fairness: Give them N samples of Obs data.
    
    metrics = {"ISD-CP": [], "PC": [], "NOTEARS": [], "Random": []}
    
    for i in range(num_graphs):
        # Generate Graph + Data manually to extract Obs Data
        pipeline_data = gen.generate_pipeline(
            num_nodes=test_vars, 
            edge_prob=0.2, 
            num_samples_base=200, # 200 Obs samples for PC/Notears
            num_samples_per_intervention=200, # Matched to ensure base_tensor is 200
            as_torch=True
        )
        
        # True Graph
        true_adj = nx.to_numpy_array(pipeline_data['dag'])
        true_adj_torch = torch.tensor(true_adj).float()
        
        # Observational Data (Base Tensor)
        # base_tensor: (num_samples_base, num_vars)
        X_obs = pipeline_data['base_tensor'] # PyTorch tensor
        X_np = X_obs.numpy()
        
        # 1. Evaluate PC
        try:
            pc_adj = run_pc(X_np)
            pc_shd = compute_shd(torch.tensor(pc_adj).unsqueeze(0), true_adj_torch.unsqueeze(0))
            metrics["PC"].append(pc_shd)
        except Exception as e:
            print(f"PC Failed: {e}")
            metrics["PC"].append(np.nan)

        # 2. Evaluate NOTEARS
        try:
            nt_adj = run_notears(X_obs, "cpu")
            nt_shd = compute_shd(torch.tensor(nt_adj).unsqueeze(0), true_adj_torch.unsqueeze(0))
            metrics["NOTEARS"].append(nt_shd)
        except Exception as e:
            print(f"NOTEARS Failed: {e}")
            metrics["NOTEARS"].append(np.nan)
            
        # 3. Evaluate Random
        rand_adj = (torch.randn(test_vars, test_vars) > 0).float()
        rand_shd = compute_shd(rand_adj.unsqueeze(0), true_adj_torch.unsqueeze(0))
        metrics["Random"].append(rand_shd)
        
        # 4. Evaluate ISD-CP
        # Needs different input format (Batches of Pairs)
        # We constructed 'pipeline_data', we can wrap it or just call model.
        # Minimal dummy call:
        # Base Samples, Intervened Samples...
        # We need Interventional Data for ISD-CP to work well (it's trained on it).
        # We generate a small batch for it.
        dataset = CausalDataset(
            gen, num_nodes_range=(test_vars, test_vars),
            samples_per_graph=32, validation_graphs=1, infinite=False
        )
        # Override the generator inside dataset? No, CausalDataset calls generator.
        # We can't easily inject the *exact* same graph into CausalDataset unless we seed/mock.
        # Hack: run CausalDataset loop until valid graph?
        # Better: Adapt inputs manually.
        pass 
        # Actually, let's just create a CausalDataset, get the batch, and extract the True Adj from it.
        # This graph will be DIFFERENT from the one above unless seeded identically.
        # FIX: We should use the SAME graph for all.
        # The 'gen.generate_pipeline' above generated a graph.
        # We can pass THAT graph to everyone.
        
        # PC/NOTEARS used X_obs from 'pipeline_data'.
        # true_adj came from 'pipeline_data'.
        # For ISD-CP, we need to format (Base, Int, Target) from 'pipeline_data'.
        # pipeline_data has 'all_dfs', 'all_masks'.
        # Let's manually collate one batch for ISD-CP.
        
        # Form Batch
        base_samples = pipeline_data['base_tensor'][:32] # (32, N)
        # Pick an intervention
        int_df = pipeline_data['all_dfs'][1] # First intervention
        int_mask = pipeline_data['all_masks'][1][0]
        int_samples = torch.tensor(int_df.values, dtype=torch.float32)[:32]
        
        target_row = base_samples
        int_mask_t = torch.tensor(int_mask).float().unsqueeze(0).expand(32, -1)
        int_idx = torch.argmax(int_mask_t, dim=1)
        
        print(f"DEBUG: base={base_samples.shape}, int={int_samples.shape}, target={target_row.shape}, mask={int_mask_t.shape}")
        
        # ISD-CP expects batches.
        # We process ONE intervention setup as a batch of 1 sample?
        # Or effectively 32 samples from the same graph?
        # The model processes (B, N) samples.
        # Our base_samples is (32, N).
        # int_samples is (32, N).
        # target_row is (32, N).
        
        # We need to ensure shapes are correct.
        # Model forward expects: (base, int, target, mask)
        
        with torch.no_grad():
            _, logits, _, _, _ = model(
                base_samples,     # (32, N)
                int_samples,      # (32, N)
                target_row,       # (32, N)
                int_mask_t        # (32, N)
            )
            # Logits: (32, N, N). Average them.
            avg_logits = logits.mean(dim=0)
            isd_shd = compute_shd(avg_logits.unsqueeze(0), true_adj_torch.unsqueeze(0))
            metrics["ISD-CP"].append(isd_shd)

    # Report
    print(f"\nResults (Mean SHD over {num_graphs} graphs):")
    for k, v in metrics.items():
        print(f"{k}: {np.nanmean(v):.2f}")

if __name__ == "__main__":
    evaluate_baselines("final_chekpoint/checkpoint_epoch_253.pt")
