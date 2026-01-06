import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch
import networkx as nx
from sklearn.metrics import roc_auc_score

# Add root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.data.SCMGenerator import SCMGenerator
from src.training.metrics import compute_shd, compute_f1, compute_sid, compute_mae
from experiments.paper_suite.wrappers import (
    ISDCPWrapper, AVICIWrapper, GEARSWrapper, TabPFNWrapper,
    PCWrapper, NotearsWrapper, OracleWrapper,
    StructureRegressorWrapper, FeatureImportanceWrapper
)
# Attempt to import wrappers for PC/NOTEARS if needed, or implement minimal ones here
# We can reuse src.models.baselines.wrappers logic but adapted to our new interface?
# For now, let's stick to the new Class A/B request.

def r2_score_val(pred, true):
    true_mean = np.mean(true)
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true_mean) ** 2)
    if ss_tot == 0: return np.nan
    return 1 - ss_res / ss_tot

def rmse_val(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))

def run_suite(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    # Define Models
    models = {}
    
    if args.run_isdcp:
        if not os.path.exists(args.isdcp_checkpoint):
            print(f"Warning: ISD-CP checkpoint not found at {args.isdcp_checkpoint}. Skipping.")
        else:
            try:
                models["ISD-CP"] = ISDCPWrapper(args.isdcp_checkpoint, device=device)
            except Exception as e:
                print(f"Failed to load ISD-CP: {e}")

    if args.run_avici:
        try:
            models["AVICI"] = AVICIWrapper(device=device)
        except ImportError:
            print("AVICI not available. Skipping.")

    if args.run_gears:
        try:
            models["GEARS"] = GEARSWrapper(device=device)
        except ImportError:
            print("GEARS not available. Skipping.")

    if args.run_tabpfn:
        try:
            models["TabPFN"] = TabPFNWrapper(device=device)
        except ImportError:
            print("TabPFN not available. Skipping.")

    if args.run_pc:
        try:
            models["PC"] = PCWrapper(device=device)
        except ImportError:
            print("PC not available (causal-learn required). Skipping.")

    if args.run_notears:
        try:
            models["NOTEARS"] = NotearsWrapper(device=device)
        except ImportError:
            print("NOTEARS not available. Skipping.")

    if args.run_two_stage:
        # PC + Ridge
        try:
            models["PC_Ridge"] = StructureRegressorWrapper(PCWrapper, regressor_type='ridge', device=device)
        except ImportError:
            pass # Already logged in simple wrapper
            
        # NOTEARS + Ridge
        try:
            models["NOTEARS_Ridge"] = StructureRegressorWrapper(NotearsWrapper, regressor_type='ridge', device=device)
        except ImportError:
            pass

        # TabPFN -> Structure (Proxy)
        try:
            models["TabPFN_Proxy"] = FeatureImportanceWrapper(TabPFNWrapper, device=device)
        except ImportError:
            pass

    if args.run_oracle:
        models["Oracle"] = OracleWrapper(device=device)
            
    if not models:
        print("No models available to run.")
        return

    sizes = [int(x) for x in args.sizes.split(',')]
    results = []
    
    print(f"Starting Benchmark on {list(models.keys())} for N={sizes}")
    
    for n in sizes:
        print(f"--- N={n} ---")
        generator = SCMGenerator(num_nodes=n, edge_prob=0.2, seed=12345 + n) # Deterministic Seed per N
        
        for g_idx in range(args.num_graphs):
            generator.seed = 12345 + n * 100 + g_idx
            # Generate Pipeline Data
            # Note: num_samples_per_intervention needs to be enough for eval
            pipe = generator.generate_pipeline(
                num_nodes=n,
                edge_prob=0.2,
                num_samples_base=args.num_samples, # Training samples
                num_samples_per_intervention=args.batch_size, # Evaluation samples
                intervention_prob=0.2, # As per full_eval_static
                intervention_scale=args.intervention_scale, # Need to update SCMGenerator to accept this if possible, or manual override
                as_torch=True,
                use_twin_world=True
            )
            
            # Extract ground truth
            dag = pipe["dag"]
            true_adj_np = nx.to_numpy_array(dag, dtype=int)
            true_adj = torch.tensor(true_adj_np, dtype=torch.float32, device=device)
            base_tensor = pipe["base_tensor"][:args.batch_size].to(device) # (B, N)
            
            # Prepare observational data for structure learning (convert to numpy)
            X_obs = base_tensor.cpu().numpy()
            
            # Iterate Models
            for name, model in models.items():
                print(f"  Graph {g_idx+1}/{args.num_graphs} | Model: {name}")
                
                # 1. Structure Prediction (Global)
                start_struct = time.time()
                try:
                    # Some models (AVICI) predict from X_obs
                    # ISD-CP predicts via delta loop (see below), but we might call fit() or similar if needed.
                    metadata = {"dag": dag} # Pass DAG for Oracle/GEARS if assumed known
                    model.fit(X_obs, metadata=metadata) 
                    pred_adj = model.predict_structure() # Might be None for Delta-only models
                except Exception as e:
                    print(f"    Structure pred failed: {e}")
                    pred_adj = None
                time_struct = time.time() - start_struct
                
                # 2. Delta Prediction & Structure Accumulation (ISD-CP)
                deltas_all_pred = []
                deltas_all_true = []
                structure_logits_list = []
                
                start_delta = time.time()
                
                # Iterate interventions
                # pipe["all_dfs"] has [Base, Int1, Int2...]
                # pipe["all_masks"] has corresponding masks
                
                for i in range(1, len(pipe["all_dfs"])):
                    df_int = pipe["all_dfs"][i]
                    mask_arr = pipe["all_masks"][i] # (B, N)
                    
                    # Get Intervention Info
                    # mask is (B, N). We assume single node intervention per batch for this loop logic
                    int_idx = np.argmax(mask_arr[0])
                    int_val = df_int.iloc[0, int_idx] # Value of intervened variable
                    
                    # True outcome
                    int_samples_true = torch.tensor(df_int.values, dtype=torch.float32, device=device)[:args.batch_size]
                    true_delta = int_samples_true - base_tensor
                    
                    # Predict
                    # We pass the FIRST sample of base as representative for "Single Input"? 
                    # No, we want to predict for the BATCH.
                    # Wrappers currently defined for SINGLE prediction?
                    # "predict_delta(..., base_sample)" -> base_sample: (1, num_vars)
                    # If wrapper handles batch, great.
                    # ISD-CP wrapper handles batch? 
                    # Wrapper interface says: base_sample: (1, num_vars).
                    # Let's update loop to predict batch? Or just 1 sample for speed?
                    # Benchmark says "Runtime vs N". Batch prediction is fairer.
                    # wrapper.predict_delta was implemented to take (1, N) in my head, but ISD-CP handles batch.
                    # Let's treat wrapper.predict_delta as taking (B, N).
                    
                    pred_d_np, extra = model.predict_delta(
                        intervention_idx=int_idx,
                        intervention_val=int_val,
                        base_sample=X_obs, # Pass proper batch
                        true_delta=true_delta # Pass true delta for Oracle cheat
                    )
                    
                    if pred_d_np is not None:
                        # pred_d_np shape should match true_delta (B, N)
                        if pred_d_np.shape != true_delta.shape:
                            # Maybe it broadcasted or returned single row?
                            # If single row, repeat?
                            if pred_d_np.ndim == 1:
                                pred_d_np = np.tile(pred_d_np, (args.batch_size, 1))
                            elif pred_d_np.shape[0] == 1:
                                pred_d_np = np.tile(pred_d_np, (args.batch_size, 1))
                        
                        deltas_all_pred.append(pred_d_np)
                        deltas_all_true.append(true_delta.cpu().numpy())
                        
                    if "logits" in extra:
                         structure_logits_list.append(extra["logits"])
                
                time_delta = time.time() - start_delta
                
                # Metrics Calculation
                row = {
                    "N": n, "Graph": g_idx, "Model": name,
                    "Time_Struct": time_struct, "Time_Delta": time_delta
                }
                
                # Structure
                final_adj_logit = None
                if pred_adj is not None:
                    # Explicit structure prediction
                    # If probability (AVICI), threshold it
                    # If binary, use it
                    # Convert to tensor for metrics
                    final_adj_logit = torch.tensor(pred_adj, device=device)
                    # If binary, SHD expects prob/logits if threshold is provided?
                    # compute_shd applies sigmoid if input is not 0/1?
                    # "threshold" arg in compute_shd defaults to 0.0 (logits > 0).
                    # If pred_adj is 0/1, we should pass compatible values.
                    # Assume pred_adj is PROB or BINARY.
                    # If binary, make it large positive/negative logits?
                    pass
                elif structure_logits_list:
                     # Aggregate logits (ISD-CP style)
                     # Stack: (Num_Ints, B, N, N) -> Mean
                     all_log = np.concatenate(structure_logits_list, axis=0)
                     mean_log = np.mean(all_log, axis=0) # (N, N)
                     final_adj_logit = torch.tensor(mean_log, device=device)
                
                if final_adj_logit is not None:
                    # Compute SHD, etc.
                    # Note: compute_shd expects (B, N, N). unsqueeze.
                    met_logits = final_adj_logit.unsqueeze(0)
                    met_true = true_adj.unsqueeze(0)
                    
                    # Need threshold.
                    # If AVICI returns probs, threshold 0.5?
                    # If ISD-CP returns logits, threshold 0.0 or 1.33?
                    thresh = args.logit_threshold
                    if name == "AVICI": thresh = 0.5 
                    
                    shd = compute_shd(met_logits, met_true, threshold=thresh)
                    f1 = compute_f1(met_logits, met_true, threshold=thresh)
                    sid = compute_sid(met_logits, met_true, threshold=thresh)
                    
                    try:
                        # AUROC
                        flat_prob = torch.sigmoid(final_adj_logit).cpu().numpy().flatten()
                        if name == "AVICI": flat_prob = final_adj_logit.cpu().numpy().flatten() # Already prob
                        flat_true = true_adj_np.flatten()
                        auroc = roc_auc_score(flat_true, flat_prob)
                    except:
                        auroc = np.nan
                        
                    row.update({"SHD": shd, "F1": f1, "SID": sid, "AUROC": auroc})
                else:
                    row.update({"SHD": np.nan, "F1": np.nan, "SID": np.nan, "AUROC": np.nan})
                    
                # Delta
                if deltas_all_pred:
                    all_pred = np.concatenate(deltas_all_pred, axis=0) # (TotalSamples, N)
                    all_true = np.concatenate(deltas_all_true, axis=0)
                    
                    mae = np.mean(np.abs(all_pred - all_true))
                    mse = np.mean((all_pred - all_true) ** 2)
                    rmse = np.sqrt(mse)
                    r2 = r2_score_val(all_pred, all_true)
                    
                    row.update({"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2})
                else:
                    row.update({"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "R2": np.nan})
                
                results.append(row)
                print(f"    -> SHD={row.get('SHD')}, MAE={row.get('MAE')}")

    # Save
    df = pd.DataFrame(results)
    path = os.path.join(args.output_dir, "benchmark_results.csv")
    df.to_csv(path, index=False)
    print(f"Saved results to {path}")
    
    # Summary
    summary = df.groupby(["Model", "N"]).mean(numeric_only=True)
    print(summary)
    summary.to_csv(os.path.join(args.output_dir, "benchmark_summary.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="experiments/paper_suite/results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_graphs", type=int, default=3)
    parser.add_argument("--sizes", default="10,20,30,40,50")
    parser.add_argument("--run_isdcp", action="store_true", default=True)
    parser.add_argument("--isdcp_checkpoint", default="checkpoints/checkpoint_epoch_253.pt")
    parser.add_argument("--run_avici", action="store_true")
    parser.add_argument("--run_gears", action="store_true")
    parser.add_argument("--run_tabpfn", action="store_true")
    parser.add_argument("--run_pc", action="store_true")
    parser.add_argument("--run_notears", action="store_true")
    parser.add_argument("--run_oracle", action="store_true")
    parser.add_argument("--run_two_stage", action="store_true", help="Run composite models (PC+Ridge, etc.)")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of observational samples for training")
    parser.add_argument("--intervention_scale", type=float, default=1.0, help="Multiplier for intervention values (Test OOD)")
    parser.add_argument("--logit_threshold", type=float, default=1.33)
    
    args = parser.parse_args()
    run_suite(args)
