import torch
import torch.nn as nn
from src.data.CausalDistributionEncoder import CausalDistributionEncoder

class ModelE_HyperNet(nn.Module):
    def __init__(self, num_nodes, d_model=128):
        super().__init__()
        self.num_nodes = num_nodes
        self.encoder = CausalDistributionEncoder(num_nodes, d_model)
        
        self.backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=4
        )
        
        # Small Hyper-Network that takes the [int_node_id] as input
        self.int_embedding = nn.Embedding(num_nodes, 16)
        self.hyper_net = nn.Sequential(
            nn.Linear(16, 64),
            nn.GELU(),
            nn.Linear(64, d_model * 1) # Outputs the weights for a final projection
        )
        
        self.dag_parent = nn.Linear(d_model, d_model)
        self.dag_child = nn.Linear(d_model, d_model)

    def forward(self, base_samples, int_samples, target_row, int_mask, int_node_idx):
        # 1. Backbone
        x = self.encoder(base_samples, int_samples, target_row, int_mask)
        x = self.backbone(x) # (B, N, d_model)
        
        # 2. Hyper-Predicted Delta Projection
        # Generate an 'Instruction Vector' based on which node was intervened
        instr = self.int_embedding(int_node_idx) # (B, 16)
        weights = self.hyper_net(instr).view(-1, 1, x.shape[-1]) # (B, 1, d_model)
        
        # Standard dynamic delta prediction: dot product of token features with hyper-weights
        deltas = torch.sum(x * weights, dim=-1) # (B, N)
        
        # 3. DAG
        p = self.dag_parent(x)
        c = self.dag_child(x)
        adj_logits = torch.matmul(p, c.transpose(-2, -1))
        
        return deltas, adj_logits