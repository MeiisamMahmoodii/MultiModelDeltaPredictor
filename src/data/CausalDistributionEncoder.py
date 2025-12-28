import torch
import torch.nn as nn

class CausalDistributionEncoder(nn.Module):
    def __init__(self, num_nodes, d_model):
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_proj = nn.Linear(6, d_model)
        self.pos_emb = nn.Embedding(num_nodes, d_model)
    
    def forward(self, base_samples, int_samples, target_row, int_mask):
        b_mean = base_samples.mean(dim=1)
        b_std = base_samples.std(dim=1) + 1e-6
        
        i_mean = int_samples.mean(dim=1)
        i_std = int_samples.std(dim=1) + 1e-6
        
        norm_i_mean = (i_mean - b_mean) / b_std
        norm_i_std = i_std / b_std
        norm_target = (target_row - b_mean) / b_std
        
        node_features = torch.stack([
            b_mean,
            b_std,
            norm_i_mean,
            norm_i_std,
            norm_target,
            int_mask
        ], dim=-1)
        
        x = self.feature_proj(node_features)
        # Add pos emb
        dev = x.device
        # Use simple broadcasting if num_nodes matches, else we assume batch matches
        # ideally we pass num_nodes or rely on tensor shape
        # But wait, self.pos_emb is fixed size? 
        # If we train with variable nodes, we must be careful.
        # usually we init encoder with MAX nodes (e.g. 50 or 100) or handle dynamic.
        # Notebook code used fixed num_nodes in constructor...
        # For variable nodes, we usually need a MAX_NODES constant.
        # But let's stick to the code we had:
        
        pos = torch.arange(x.shape[1], device=dev).unsqueeze(0) # (1, N)
        # If model expects up to 100 nodes but current batch has 20, we slice or use embedding
        x = x + self.pos_emb(pos)
        return x
