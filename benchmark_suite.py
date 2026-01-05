import argparse
import os
import time
import pandas as pd
import numpy as np
import networkx as nx
import torch
from src.data.SCMGenerator import SCMGenerator
from src.training.metrics import compute_shd, compute_f1, compute_tpr_fdr
from src.models.baselines.wrappers import PCWrapper, GESWrapper, NotearsWrapper, ISDCPWrapper
# Import other wrappers if implemented
# from src.models.baselines.wrappers import DAGGNNWrapper, GraNDAGWrapper

def run_benchmark(max_vars_list=[10, 20, 30, 50, 100], 
                  densities=[0.1, 0.2, 0.3], 
                  models_to_run=['PC', 'ISD-CP'],
                  device='cpu',
                  output_file='benchmark_comprehensive.csv',
                  dry_run=False):
    
    results = []
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or '.', exist_ok=True)
    
    print(f"--- Starting Comprehensive Benchmark ---")
    print(f"Models: {models_to_run}")
    print(f"Scales: {max_vars_list}")
    print(f"Densities: {densities}")
    
    for n_vars in max_vars_list:
        for density in densities:
            print(f"\n[Config] Nodes: {n_vars}, Density: {density}")
            
            # 1. Generate Standard Benchmark Dataset
            # Fixed seed for reproducibility implies we re-seed SCMGenerator or use fixed set
            # For simplicity, we generate ONE large graph per config for this study
            # In a real paper, we'd average over 10 seeds.
            
            seed = 42
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            gen = SCMGenerator(num_nodes=n_vars, edge_prob=density, noise_scale=1.0)
            # Generate graph structure (force sampled)
            # SCMGenerator generates structure lazily per sample? 
            # Actually __init__ often sets up the graph if fixed? No, SCMGenerator creates random DAGs.
            # We need a FIXED structure for benchmarking.
            # We need to extract the adjacency matrix from the generator or create one manually.
            # Looking at SCMGenerator: it usually generates a sample and graph together or we can ask for a specific DAG.
            
            # Let's interact with SCMGenerator basic usage:
            # We want to sample N samples from ONE DAG.
            # SCMGenerator.sample(params) -> creates graph if not fixed?
            # Let's assume we can generate a dag manually.
            
            # Generate DAG and Data explicitly
            dag = gen.generate_dag(num_nodes=n_vars, edge_prob=density, seed=seed)
            dag = gen.edge_parameters(dag)
            
            # Generate observational data (1000 samples)
            samples_data_df, _ = gen.generate_data(dag, num_samples=1000)
            
            # Convert to numpy for wrappers
            X = samples_data_df.values
            true_adj = nx.to_numpy_array(dag, nodelist=range(n_vars), weight=None)
            # Binary adjacency for evaluation
            true_adj = (true_adj != 0).astype(int)
            
            # 2. Run Models
            for model_name in models_to_run:
                print(f"  -> Running {model_name}...", end='', flush=True)
                start_time = time.time()
                
                try:
                    if model_name == 'PC':
                        model = PCWrapper(alpha=0.01)
                    elif model_name == 'GES':
                        model = GESWrapper()
                    elif model_name == 'NOTEARS':
                        model = NotearsWrapper(use_gpu=(device!='cpu'))
                    elif model_name == 'ISD-CP':
                        model = ISDCPWrapper(device=device, d_model=256) # Configurable
                    elif model_name == 'DAG-GNN':
                         # Placeholder for Tier 1
                         print(" (Skipped/Not Impl)", end='')
                         continue
                    else:
                        print(f" Unknown model {model_name}", end='')
                        continue
                        
                    # FIT
                    model.fit(X)
                    
                    # PREDICT
                    pred_adj = model.predict_adj()
                    
                    # METRICS
                    runtime = time.time() - start_time
                    
                    # Convert to tensor for existing metrics
                    pred_t = torch.tensor(pred_adj, dtype=torch.float32)
                    true_t = torch.tensor(true_adj, dtype=torch.float32)
                    
                    shd = compute_shd(pred_t.unsqueeze(0), true_t.unsqueeze(0)) # expects batch?
                    f1 = compute_f1(pred_t.unsqueeze(0), true_t.unsqueeze(0))
                    tpr, fdr = compute_tpr_fdr(pred_t.unsqueeze(0), true_t.unsqueeze(0))
                    
                    print(f" Done ({runtime:.2f}s) | SHD: {shd:.1f} | F1: {f1:.2f}")
                    
                    results.append({
                        'Nodes': n_vars,
                        'Density': density,
                        'Model': model_name,
                        'SHD': shd,
                        'F1': f1,
                        'TPR': tpr,
                        'FDR': fdr,
                        'Time': runtime
                    })
                    
                except Exception as e:
                    print(f" Failed: {e}")
                    results.append({
                        'Nodes': n_vars,
                        'Density': density,
                        'Model': model_name,
                        'SHD': -1,
                        'F1': -1,
                        'TPR': -1,
                        'FDR': -1,
                        'Time': -1,
                        'Error': str(e)
                    })
                
                # Save after every model to prevent data loss
                pd.DataFrame(results).to_csv(output_file, index=False)
                
            if dry_run: 
                print("Dry Run: Breaking after first config.")
                break
        if dry_run: break
        
    print(f"\nBenchmark Complete. Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_vars', type=str, default="10,20,30,50,100", help="Comma separated list of node kinds")
    parser.add_argument('--models', type=str, default="PC,GES,NOTEARS,ISD-CP", help="Comma separated models")
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    vars_list = [int(x) for x in args.max_vars.split(',')]
    models = args.models.split(',')
    
    run_benchmark(max_vars_list=vars_list, models_to_run=models, dry_run=args.dry_run, device=args.device)
