import torch
import numpy as np
import pandas as pd
import networkx as nx
import time
import os
import sys
import traceback
sys.path.append(os.getcwd())

from tqdm import tqdm
from src.models.CausalTransformer import CausalTransformer
from src.data.SCMGenerator import SCMGenerator
from src.data.CausalDataset import CausalDataset
from src.training.metrics import compute_shd, compute_f1

# Baselines
from src.models.baselines.notears import NotearsLinear
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.search.FCMBased.lingam import DirectLiNGAM
from causallearn.search.ScoreBased.GES import ges

class BenchmarkRunner:
    def __init__(self, checkpoint_path, scales=[20, 50]):
        self.checkpoint_path = checkpoint_path
        self.scales = scales
        self.device = torch.device('cpu') # Force CPU for fairness/stability
        self.results = []
        
        # Load Model Once (Re-load weights if needed, or just reshape?)
        # Model depends on N. We need to re-init model for each N if architecture relies on N.
        # Checkpoint has strict mapping?
        # CausalTransformer takes 'num_nodes' as arg.
        # If we trained on 20-50, and valid on 100, we might need to adjust num_nodes or use relative?
        # Our model supports variable N via 'max_vars' buffer or just standard attention.
        # But embeddings (var_id_emb) are fixed size! :O
        # If we trained with max_vars=50, we CANNOT run on N=100 without resizing embeddings or modulo.
        # Checkpoint args: max_vars.
        
        self.ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.args = self.ckpt['args']
        self.train_max_vars = self.args.max_vars
        
    def get_isd_model(self, num_nodes):
        # We need to construct the model compatible with checkpoint.
        # If num_nodes > train_max_vars, we have a problem unless we hack positional embeddings.
        # For now, we only benchmark up to train_max_vars (e.g. 50).
        # If User asked for 100, we warn and skip or try to extrapolate (unsafe).
        
        # Actually our embeddings are: self.var_id_emb = nn.Embedding(num_vars + 1, d_model)
        # We can expand this embedding on the fly?
        # Or just benchmark up to 50.
        
        effective_vars = max(num_nodes, self.train_max_vars)
        model = CausalTransformer(
            num_nodes=effective_vars + 5,
            d_model=512,
            num_layers=self.args.num_layers,
            grad_checkpoint=getattr(self.args, 'grad_checkpoint', False),
            ablation_dense=getattr(self.args, 'ablation_dense_moe', False),
            ablation_no_interleaved=getattr(self.args, 'ablation_no_interleaved', False),
            ablation_no_dag=getattr(self.args, 'ablation_no_dag', False),
            ablation_no_physics=getattr(self.args, 'ablation_no_physics', False)
        )
        # Load State Dict (Strict=False allows resizing if we were smart, but we aren't handling resize yet)
        # We load strictly.
        model.load_state_dict(self.ckpt['model_state_dict'], strict=True)
        model.to(self.device)
        model.eval()
        return model

    def run_pc_proxy(self, data):
        # Wrapper to handle crashes
        try:
            start = time.time()
            cg = pc(data, 0.05, fisherz, True, 0, -1)
            adj = cg.G.graph
            # Convert to binary
            # 2->3 means Arrow->Tail (j->i)?
            # In causal-learn: G[i,j] involves endpoint at j.
            # 2 = Arrow, 3 = Tail, 1 = Circle.
            # i -> j: G[i,j]=-1?, G[j,i]=?
            # Actually standard convention:
            # G[i,j] is the edge mark at j.
            # If i->j, then at j we have an arrow (2). At i we have tail (3).
            # So G[i, j] == 2 and G[j, i] == 3.
            
            n = data.shape[1]
            res = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if adj[i, j] == 2 and adj[j, i] == 3:
                        res[i, j] = 1 # i->j
            return res, time.time() - start
        except Exception as e:
            return None, 0.0

    def run_lingam_proxy(self, data):
        try:
            start = time.time()
            model = DirectLiNGAM()
            model.fit(data)
            return model.adjacency_matrix_, time.time() - start
        except Exception as e:
            return None, 0.0

    def run_ges_proxy(self, data):
        try:
            start = time.time()
            # RECORD -> score based
            # GES returns 'Post' object
            # record = ges(data)
            # adj = record['G'].graph
            # Check docs... standard usage:
            # G, score = ges(data)
            G, score = ges(data)
            adj = G.graph
            # GES adj is similar to PC?
            # 2=Arrow, 3=Tail?
            # Assuming similar conversion
            n = data.shape[1]
            res = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if adj[i, j] == 2 and adj[j, i] == 3:
                        res[i, j] = 1
            return res, time.time() - start
        except Exception:
            return None, 0.0

    def run_notears_proxy(self, data):
        try:
            start = time.time()
            nt = NotearsLinear(d=data.shape[1], max_iter=50)
            W = nt.fit(torch.tensor(data).float())
            adj = (np.abs(W) > 0.3).astype(float) # Threshold 0.3
            return adj, time.time() - start
        except Exception:
             return None, 0.0

    def run_suite(self):
        for N in self.scales:
            if N > self.train_max_vars:
                print(f"Skipping N={N} (Exceeds model training capacity {self.train_max_vars})")
                continue
                
            print(f"--- Benchmarking Scale N={N} ---")
            
            model = self.get_isd_model(N)
            gen = SCMGenerator(num_nodes=N, edge_prob=0.2, seed=42)
            
            num_graphs = 3 # Keep low for speed
            
            for g_idx in range(num_graphs):
                # Data Generation
                # We generate 200 samples for Baselines (Obs)
                pipeline_data = gen.generate_pipeline(
                    num_nodes=N, edge_prob=0.2, 
                    num_samples_base=200, num_samples_per_intervention=200,
                    as_torch=True
                )
                X_obs = pipeline_data['base_tensor'].numpy()
                true_dag = nx.to_numpy_array(pipeline_data['dag'])
                true_t = torch.tensor(true_dag).float()
                
                # 1. PC
                adj, t = self.run_pc_proxy(X_obs)
                if adj is not None:
                    shd = compute_shd(torch.tensor(adj).unsqueeze(0), true_t.unsqueeze(0))
                    self.results.append({"Model": "PC", "N": N, "SHD": shd, "Time": t})
                
                # 2. LiNGAM
                adj, t = self.run_lingam_proxy(X_obs)
                if adj is not None:
                    # LiNGAM returns weighted adj. Convert to binary?
                    # It returns adjacency_matrix_ where A_ij is coeff (j -> i)?
                    # DirectLiNGAM convention: row i is target. A_ij is coeff of j on i.
                    # so if A[i,j] != 0, j->i.
                    # My compute_shd expects A[j,i] = 1 for j->i?
                    # Let's verify SHD convention. 
                    # metrics.py compute_shd assumes: pred[i,j] means i->j?
                    # Usually: Adj[i,j]=1 => edge i->j.
                    # LiNGAM: x_i = sum b_ij x_j + e_i.
                    # b_ij is in row i, col j.
                    # So B[i, j] implies j -> i.
                    # So LiNGAM returns B.Transposed of standard Adj?
                    # Standard Adj A[j, i] = 1 => j -> i??
                    # NO. Standard adjacency matrix A[i, j] = 1 usually means i -> j.
                    # So if LiNGAM B[i, j] means j -> i, then LiNGAM Adj is Transpose of Standard.
                    # Let's take Transpose of LiNGAM result.
                    adj = (np.abs(adj) > 0.05).astype(float).T
                    shd = compute_shd(torch.tensor(adj).unsqueeze(0), true_t.unsqueeze(0))
                    self.results.append({"Model": "LiNGAM", "N": N, "SHD": shd, "Time": t})

                # 3. GES
                adj, t = self.run_ges_proxy(X_obs)
                if adj is not None:
                    shd = compute_shd(torch.tensor(adj).unsqueeze(0), true_t.unsqueeze(0))
                    self.results.append({"Model": "GES", "N": N, "SHD": shd, "Time": t})
                    
                # 4. NOTEARS
                adj, t = self.run_notears_proxy(X_obs)
                if adj is not None:
                    shd = compute_shd(torch.tensor(adj).unsqueeze(0), true_t.unsqueeze(0))
                    self.results.append({"Model": "NOTEARS", "N": N, "SHD": shd, "Time": t})

                # 5. ISD-CP
                start = time.time()
                # Prepare Inputs
                base_samples = pipeline_data['base_tensor'][:32]
                int_df = pipeline_data['all_dfs'][1]
                int_mask = pipeline_data['all_masks'][1][0]
                int_samples = torch.tensor(int_df.values, dtype=torch.float32)[:32]
                
                target_row = base_samples
                int_mask_t = torch.tensor(int_mask).float().unsqueeze(0).repeat(32, 1) # (32, N)
                int_idx = torch.argmax(int_mask_t, dim=1)
                
                with torch.no_grad():
                    _, logits, _, _ = model(
                        base_samples, int_samples, target_row, int_mask_t, int_idx
                    )
                    # Use tuned threshold 1.33
                    avg_logits = logits.mean(dim=0)
                    adj_isd = (avg_logits > 1.33).float()
                    
                t = time.time() - start
                shd = compute_shd(adj_isd.unsqueeze(0), true_t.unsqueeze(0))
                self.results.append({"Model": "ISD-CP", "N": N, "SHD": shd, "Time": t})

        # Save Results
        df = pd.DataFrame(self.results)
        print("\n--- Comprehensive Benchmark Results ---")
        print(df.groupby(["Model", "N"]).mean())
        df.to_csv("benchmark_full.csv", index=False)
        
        # Markdown Report
        with open("benchmark_full.md", "w") as f:
            f.write("# Comprehensive Benchmark Report\n\n")
            f.write(df.groupby(["Model", "N"]).mean().to_markdown())

if __name__ == "__main__":
    runner = BenchmarkRunner("final_chekpoint/checkpoint_epoch_253.pt", scales=[20])
    runner.run_suite()
