import torch
import torch.nn as nn
from src.data.CausalDistributionEncoder import CausalDistributionEncoder

class ModelA_Baseline(nn.Module):
    def __init__(self, num_nodes, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.encoder = CausalDistributionEncoder(num_nodes, d_model)
        
        # Shared Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            activation="gelu", 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Head 1: Delta Prediction (Per token scalar)
        self.delta_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        
        # Head 2: DAG Prediction (Parent-Child Bilinear Head)
        self.dag_parent = nn.Linear(d_model, d_model)
        self.dag_child = nn.Linear(d_model, d_model)

    def forward(self, base_samples, int_samples, target_row, int_mask, int_node_idx=None):
        # 1. Encode into token sequence (Batch, Nodes, d_model)
        x = self.encoder(base_samples, int_samples, target_row, int_mask)
        
        # 2. Process via Transformer
        x = self.transformer(x)
        
        # 3. Predict Delta
        deltas = self.delta_head(x).squeeze(-1) # (Batch, Nodes)
        
        # 4. Predict DAG Adjacency Matrix
        p = self.dag_parent(x) # (B, N, d_model)
        c = self.dag_child(x)  # (B, N, d_model)
        adj_logits = torch.matmul(p, c.transpose(-2, -1)) # (B, N, N)
        
        return deltas, adj_logits