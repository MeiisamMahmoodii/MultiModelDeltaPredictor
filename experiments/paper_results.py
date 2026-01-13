import torch
import numpy as np
import pandas as pd
import time
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
sys.path.append(os.getcwd())

from tqdm import tqdm
from src.models.CausalTransformer import CausalTransformer
from src.data.SCMGenerator import SCMGenerator
from src.training.metrics import compute_shd

# Baselines
from src.models.baselines.notears import NotearsLinear
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
# Re-import LiNGAM if needed, but we focus on PC/Notears for speed in loops
from causallearn.search.FCMBased.lingam import DirectLiNGAM

class PaperExperiments:
    def __init__(self, checkpoint_path):
        self.device = torch.device('cpu') 
        self.ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.args = self.ckpt['args']
        
        # Load Model
        self.model = CausalTransformer(
            num_nodes=self.args.max_vars + 5,
            d_model=512,
            num_layers=self.args.num_layers,
            grad_checkpoint=getattr(self.args, 'grad_checkpoint', False),
            ablation_dense=getattr(self.args, 'ablation_dense_moe', False),
            ablation_no_interleaved=getattr(self.args, 'ablation_no_interleaved', False),
            ablation_no_dag=getattr(self.args, 'ablation_no_dag', False),
            ablation_no_physics=getattr(self.args, 'ablation_no_physics', False)
        )
        self.model.load_state_dict(self.ckpt['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
    def run_pc(self, data):
        try:
            start = time.time()
            cg = pc(data, 0.05, fisherz, True, 0, -1)
            adj = cg.G.graph
            n = data.shape[1]
            res = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if adj[i, j] == 2 and adj[j, i] == 3:
                        res[i, j] = 1 
            return res, time.time() - start
        except:
            return None, 0.0

    def run_isd_cp(self, pipeline_data, ensemble=1):
        start = time.time()
        
        num_samples = pipeline_data['base_tensor'].shape[0]
        slice_size = min(32, num_samples)
        
        base_samples = pipeline_data['base_tensor'][:slice_size]
        int_df = pipeline_data['all_dfs'][1]
        int_mask = pipeline_data['all_masks'][1][0]
        int_samples = torch.tensor(int_df.values, dtype=torch.float32)[:slice_size]
        
        target_row = base_samples
        int_mask_t = torch.tensor(int_mask).float().unsqueeze(0).repeat(slice_size, 1) # (B, N)
        int_idx = torch.argmax(int_mask_t, dim=1)[:slice_size]
        
        # Ensembling: Run multiple times?
        # Model is deterministic unless dropout is ON (eval turns it off).
        # We need to vary INPUTS (permutation).
        # Since transformer is order-invariant (modulo positional encoding if we had it, but we use RoPE/Set).
        # Actually RoPE IS sensitive to position.
        # So shuffling nodes WILL change output.
        
        logits_sum = None
        
        N = base_samples.shape[1]
        perm = torch.arange(N)
        
        for k in range(ensemble):
            # 1. Permute
            if k > 0:
                perm = torch.randperm(N)
                
            # Apply permutation to inputs
            perm_base = base_samples[:, perm]
            perm_int = int_samples[:, perm]
            perm_target = target_row[:, perm]
            perm_mask = int_mask_t[:, perm]
            # idx must map to new position
            # If target was at T, new position is where T moved to.
            # perm[new_pos] = old_pos
            # We want new index.
            # int_idx is vector of indices.
            # We need to find where indices went.
            # mapping: old -> new. 
            # inverse perm: inv[old] = new
            inv_perm = torch.argsort(perm)
            
            with torch.no_grad():
                # Unpack 5 values
                _, logits, _, _, _ = self.model(
                    perm_base, perm_int, perm_target, perm_mask
                )
                
            # Logits are (B, N, N) in Permuted Order.
            # We must un-permute them back to original canonical order.
            # Row i is node P[i]. Col j is node P[j].
            # We want Row k (original node k).
            # Original k is at index inv_perm[k].
            # So Unpermuted[k, :] = Permuted[inv_perm[k], :]
            # Then columns: Unpermuted[:, l] = Permuted[:, inv_perm[l]]
            
            # Smart un-permute:
            # L_orig = L_perm[inv_perm][:, inv_perm]
            
            avg_logits = logits.mean(dim=0) # (N, N)
            unperm_logits = avg_logits[inv_perm][:, inv_perm]
            
            if logits_sum is None:
                logits_sum = unperm_logits
            else:
                logits_sum += unperm_logits
                
        final_logits = logits_sum / ensemble
        adj = (final_logits > 1.33).float() # Tuned Threshold
        return adj, time.time() - start

    def exp_sample_efficiency(self):
        print("\n--- Experiment: Sample Efficiency ---")
        N = 20
        samples_list = [16, 32, 64, 128, 200]
        results = []
        gen = SCMGenerator(num_nodes=N, edge_prob=0.2, seed=42)
        
        for s in samples_list:
            print(f"Testing Samples={s}...")
            # Run 5 graphs
            for _ in range(5):
                pipe = gen.generate_pipeline(N, 0.2, num_samples_base=s, num_samples_per_intervention=s, as_torch=True)
                true_t = torch.tensor(nx.to_numpy_array(pipe['dag'])).float()
                
                # PC
                adj_pc, _ = self.run_pc(pipe['base_tensor'].numpy())
                if adj_pc is not None:
                    shd_pc = compute_shd(torch.tensor(adj_pc).unsqueeze(0), true_t.unsqueeze(0))
                    results.append({"Model": "PC", "Samples": s, "SHD": shd_pc})
                
                # ISD-CP (Ensemble=1)
                # Note: Our model is trained on varying samples?
                # Usually fixed batch size 32. If we give 16, we pad or repeat?
                # Model handles (B, N).
                # We simply pass min(32, s) or full s.
                # If s < 32, we just pass s. 
                # Model works on sets.
                adj_isd, _ = self.run_isd_cp(pipe, ensemble=1)
                shd_isd = compute_shd(adj_isd.unsqueeze(0), true_t.unsqueeze(0))
                results.append({"Model": "ISD-CP", "Samples": s, "SHD": shd_isd})
                
        df = pd.DataFrame(results)
        print(df.groupby(["Model", "Samples"]).mean())
        df.to_csv("exp_sample_efficiency.csv", index=False)
        return df

    def exp_time_scaling(self):
        print("\n--- Experiment: Time Scaling ---")
        N_list = [10, 20, 30, 40, 50]
        results = []
        
        for N in N_list:
            print(f"Testing Scale N={N}...")
            gen = SCMGenerator(num_nodes=N, edge_prob=0.2, seed=42)
            # 200 samples for fairness
            try:
                pipe = gen.generate_pipeline(N, 0.2, num_samples_base=200, num_samples_per_intervention=200, as_torch=True)
                
                # PC
                _, t_pc = self.run_pc(pipe['base_tensor'].numpy())
                results.append({"Model": "PC", "N": N, "Time": t_pc})
                
                # ISD-CP
                _, t_isd = self.run_isd_cp(pipe, ensemble=1)
                results.append({"Model": "ISD-CP", "N": N, "Time": t_isd})
            except Exception as e:
                print(f"Failed N={N}: {e}")
                
        df = pd.DataFrame(results)
        print(df.groupby(["Model", "N"]).mean())
        df.to_csv("exp_time_scaling.csv", index=False)
        return df

    def exp_ensembling(self):
        print("\n--- Experiment: Test-Time Ensembling ---")
        N = 20
        gen = SCMGenerator(num_nodes=N, edge_prob=0.2, seed=42)
        results = []
        
        for _ in range(5):
            pipe = gen.generate_pipeline(N, 0.2, num_samples_base=200, num_samples_per_intervention=200, as_torch=True)
            true_t = torch.tensor(nx.to_numpy_array(pipe['dag'])).float()
            
            # Ensemble 1 (Baseline)
            adj_1, _ = self.run_isd_cp(pipe, ensemble=1)
            shd_1 = compute_shd(adj_1.unsqueeze(0), true_t.unsqueeze(0))
            results.append({"Ensemble": 1, "SHD": shd_1})
            
            # Ensemble 5
            adj_5, _ = self.run_isd_cp(pipe, ensemble=5)
            shd_5 = compute_shd(adj_5.unsqueeze(0), true_t.unsqueeze(0))
            results.append({"Ensemble": 5, "SHD": shd_5})

            # Ensemble 10
            adj_10, _ = self.run_isd_cp(pipe, ensemble=10)
            shd_10 = compute_shd(adj_10.unsqueeze(0), true_t.unsqueeze(0))
            results.append({"Ensemble": 10, "SHD": shd_10})
            
        df = pd.DataFrame(results)
        print(df.groupby(["Ensemble"]).mean())
        df.to_csv("exp_ensembling.csv", index=False)
        return df

if __name__ == "__main__":
    exp = PaperExperiments("final_chekpoint/checkpoint_epoch_253.pt")
    
    # Run All
    exp.exp_sample_efficiency()
    exp.exp_time_scaling()
    exp.exp_ensembling()
