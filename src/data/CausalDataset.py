import torch
import numpy as np
import networkx as nx
from torch.utils.data import IterableDataset

class CausalDataset(IterableDataset):
    def __init__(self, generator, num_nodes_range=(5, 10), samples_per_graph=100, edge_prob=0.3, intervention_prob=0.5, infinite=True, validation_graphs=32, reuse_factor=1):
        self.generator = generator
        self.num_nodes_range = num_nodes_range
        self.samples_per_graph = samples_per_graph
        self.edge_prob = edge_prob
        self.intervention_prob = intervention_prob
        self.infinite = infinite
        self.validation_graphs = validation_graphs
        self.reuse_factor = reuse_factor
    
    def __iter__(self):
        graphs_generated = 0
        while True:
            if not self.infinite and graphs_generated >= self.validation_graphs:
                break
                
            n = np.random.randint(self.num_nodes_range[0], self.num_nodes_range[1] + 1)
            res = self.generator.generate_pipeline(
                num_nodes=n,
                edge_prob=self.edge_prob,
                num_samples_base=self.samples_per_graph,
                num_samples_per_intervention=self.samples_per_graph,
                intervention_prob=self.intervention_prob,
                as_torch=True
            )
            
            graphs_generated += 1
            
            # --- DATA REUSE CACHE ---
            # We want to yield this EXACT same graph and data `reuse_factor` times.
            # However, yields in Python generators are sequential.
            # We can't "rewind". 
            # So we collect all items from this graph first?
            # Or simpler: We just wrap the yield loop in a repeat loop!
            
            adj = torch.tensor(nx.to_numpy_array(res['dag']), dtype=torch.float32)
            base_tensor = res['base_tensor']
            
            # Pre-calculate interactions to avoid re-computing inside loop
            interactions = []
            for i in range(1, len(res['all_dfs'])):
                int_tensor = torch.tensor(res['all_dfs'][i].values, dtype=torch.float32)
                int_mask = torch.tensor(res['all_masks'][i][0], dtype=torch.float32)
                int_node_idx = torch.argmax(int_mask)
                interactions.append((int_tensor, int_mask, int_node_idx))
            
            # REUSE LOOP
            for _ in range(self.reuse_factor):
                for int_tensor, int_mask, int_node_idx in interactions:
                    # Twin World Matching:
                    for j in range(int_tensor.shape[0]):
                        target_row = base_tensor[j]
                        intervened_row = int_tensor[j]
                        
                        # Delta calculation:
                        delta = intervened_row - target_row
                        
                        yield {
                            "base_samples": base_tensor,
                            "int_samples": int_tensor, 
                            "target_row": target_row,
                            "int_mask": int_mask,
                            "int_node_idx": int_node_idx,
                            "delta": delta,
                            "adj": adj
                        }
