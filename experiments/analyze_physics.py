import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import os
import sys
sys.path.append(os.getcwd())

from src.models.CausalTransformer import CausalTransformer
from src.data.SCMGenerator import SCMGenerator
from src.data.CausalDataset import CausalDataset
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

def run_pc(data_np):
    cg = pc(data_np, 0.05, fisherz, True, 0, -1)
    return cg.G.graph

def get_pc_parents(adj, target_idx):
    # causal-learn adj[i, j]
    # In PC output:
    # 2 = Arrow at j (from i). i -> j.
    # 3 = Tail at i.
    # So if adj[i, target] == 2 and adj[target, i] == 3: i is parent
    n = adj.shape[0]
    parents = []
    for i in range(n):
        if i == target_idx: continue
        if adj[i, target_idx] == 2 and adj[target_idx, i] == 3:
            parents.append(i)
    return parents

def analyze_physics(checkpoint_path, num_graphs=5, test_vars=20):
    print(f"--- Analyzing Physics (N={test_vars}) ---")
    
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
    
    gen = SCMGenerator(num_nodes=test_vars, edge_prob=0.2, seed=42)
    
    metrics = {"ISD-CP": [], "Linear(All)": [], "Linear(PC)": []}
    
    for _ in range(num_graphs):
        # 1. Generate Training Data (Observational) for Baselines
        pipe_train = gen.generate_pipeline(
            num_nodes=test_vars, 
            edge_prob=0.2, 
            num_samples_base=500, # Train regressor on 500 samples
            num_samples_per_intervention=1,
            as_torch=False
        )
        
        X_train = pipe_train['base_tensor'].values
        
        # 2. Run PC to find parents (Structural Baseline)
        pc_adj = run_pc(X_train)
        
        # 3. Train Repressors per variable
        regressors_all = {}
        regressors_pc = {}
        
        for i in range(test_vars):
            # Target y = X[:, i]
            # Inputs = X[:, not i] (All) OR X[:, Parents] (PC)
            y_train = X_train[:, i]
            
            # Linear(All)
            X_all = np.delete(X_train, i, axis=1) # All except self
            reg_all = Ridge().fit(X_all, y_train)
            regressors_all[i] = reg_all
            
            # Linear(PC)
            parents = get_pc_parents(pc_adj, i)
            if not parents:
                # No parents found -> Predict Mean
                regressors_pc[i] = np.mean(y_train)
            else:
                X_pc = X_train[:, parents]
                reg_pc = Ridge().fit(X_pc, y_train)
                regressors_pc[i] = (reg_pc, parents)
                
        # 4. Evaluate on Interventional Data (OOD Physics)
        # Generate Test Batch
        pipe_test = gen.generate_pipeline(
            num_nodes=test_vars, 
            edge_prob=0.2, 
            num_samples_base=32, 
            num_samples_per_intervention=32,
            as_torch=True
        )
        # Override dag to match train?
        # NO. generate_pipeline generates NEW DAG each call unless seeded.
        # We need to train/test on SAME DAG.
        # FIX: SCMGenerator creates new DAG every call.
        # We must manually extract DAG from pipe_train and generate more data from it.
        # SCMGenerator.generate_data(dag, ...)
        
        dag = pipe_train['dag']
        
        # Generate Test Interventions on THIS dag.
        # Pick random target
        target_node = np.random.choice(test_vars)
        intervention_val = 5.0
        
        # Generate Ground Truth Counterfactuals
        # We need "Base" and "Intervened" for specific target
        # Use simple generate_data
        base_noise = np.random.normal(0, 1, (32, test_vars))
        df_base, _ = gen.generate_data(dag, 32, noise=base_noise)
        df_int, _ = gen.generate_data(dag, 32, noise=base_noise, intervention={target_node: intervention_val})
        
        # True Delta
        true_delta = df_int.values - df_base.values # (32, N)
        
        # A. ISD-CP Prediction
        base_t = torch.tensor(df_base.values).float()
        int_t = torch.tensor(df_int.values).float()
        target_row = base_t # Context
        
        mask = torch.zeros(32, test_vars)
        mask[:, target_node] = 1.0
        mask_t = mask.float()
        idx_t = torch.argmax(mask_t, dim=1)
        
        with torch.no_grad():
            deltas_pred, _, _, _ = model(base_t, int_t, target_row, mask_t, idx_t)
            # MAE
            mae_isd = mean_absolute_error(true_delta, deltas_pred.numpy())
            metrics["ISD-CP"].append(mae_isd)
            
        # B. Linear Baselines Prediction
        # Delta = Regressor(X_int) - Regressor(X_base)? 
        # Or Regressor(X_base_with_intervention)?
        # Ideally: Physics Predictor should predict X_new given (X_old, do(T=v)).
        # Interventional Prediction.
        # Ours predicts Delta directly.
        # Baselines: Predict X_int then subtract X_base.
        
        preds_all = np.zeros_like(df_int.values)
        preds_pc = np.zeros_like(df_int.values)
        
        for i in range(test_vars):
            if i == target_node:
                preds_all[:, i] = intervention_val
                preds_pc[:, i] = intervention_val
                continue
            
            # Predict X[i] given Intervened Context
            # Context for prediction is df_int (where target is set)
            # But we don't know the downstream effects yet!
            # We only know inputs.
            # Causal Prediction: We have values of parents.
            # If parents are upstream of target, they are unchanged (same as base).
            # If parents include target, it is changed.
            # So we use df_int (ground truth parents) ??
            # NO. That leaks the answer.
            # We must use "Computed" values.
            # Order matters!
            # But Regressors don't know order.
            # Standard "Do-Calculus" approximation with regression:
            # Predict using observed values.
            # If we simply plug in the interventional vector (with target modified) into the regressor,
            # does it predict the correct downstream value?
            # Yes, if the regressor learned the structural equation X_i = f(Parents).
            # If it learned X_i = f(Children) (Reverse causality), it will fail OOD.
            # This is exactly what we want to test!
            
            # Input vector for prediction:
            # We use df_base values for non-targets?
            # Or do we use the "Intervened" vector where T=val?
            # We use X_base but replace Target column with Val.
            X_input = df_base.values.copy()
            X_input[:, target_node] = intervention_val
            
            # Linear(All)
            X_in_all = np.delete(X_input, i, axis=1)
            preds_all[:, i] = regressors_all[i].predict(X_in_all)
            
            # Linear(PC)
            if isinstance(regressors_pc[i], tuple):
                reg, parents = regressors_pc[i]
                X_in_pc = X_input[:, parents]
                preds_pc[:, i] = reg.predict(X_in_pc)
            else:
                preds_pc[:, i] = regressors_pc[i] # Mean
                
        # Calculate Delta from Predictions
        delta_all = preds_all - df_base.values
        delta_pc = preds_pc - df_base.values
        
        metrics["Linear(All)"].append(mean_absolute_error(true_delta, delta_all))
        metrics["Linear(PC)"].append(mean_absolute_error(true_delta, delta_pc))

    print("\nPhysics Analysis Results (MAE):")
    for k, v in metrics.items():
        print(f"{k}: {np.mean(v):.4f}")

if __name__ == "__main__":
    analyze_physics("final_chekpoint/checkpoint_epoch_253.pt")
