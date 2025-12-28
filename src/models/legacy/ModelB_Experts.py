import torch
import torch.nn as nn
from src.data.CausalDistributionEncoder import CausalDistributionEncoder

class ModelB_Experts(nn.Module):
    def __init__(self, num_nodes, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.encoder = CausalDistributionEncoder(num_nodes, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Head 1: VARIABLE-SPECIFIC Experts (using a ModuleList for distinct parameters per node)
        self.delta_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1)
            ) for _ in range(num_nodes)
        ])
        
        # Head 2: DAG Prediction
        self.dag_parent = nn.Linear(d_model, d_model)
        self.dag_child = nn.Linear(d_model, d_model)

    def forward(self, base_samples, int_samples, target_row, int_mask, int_node_idx=None):
        x = self.encoder(base_samples, int_samples, target_row, int_mask)
        x = self.transformer(x) # (B, N, d_model)
        
        # Apply each expert to its corresponding token
        deltas = []
        # Use actual sequence length from the batch, not the max capacity
        # We assume x.shape[1] <= self.num_nodes (enforced by dataset range)
        current_num_nodes = x.shape[1]
        for i in range(current_num_nodes):
            # x[:, i, :] is the token for node i
            d_i = self.delta_experts[i](x[:, i, :]) # (B, 1)
            deltas.append(d_i)
        deltas = torch.cat(deltas, dim=-1) # (B, N)
        
        p = self.dag_parent(x)
        c = self.dag_child(x)
        adj_logits = torch.matmul(p, c.transpose(-2, -1))
        
        return deltas, adj_logits