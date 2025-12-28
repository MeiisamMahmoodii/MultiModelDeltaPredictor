import torch
import torch.nn as nn
from src.data.CausalDistributionEncoder import CausalDistributionEncoder

def gumbel_sigmoid(logits, tau=1.0, hard=False):
    """Differentiable sampling for binary edges."""
    gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.sigmoid()

    if hard:
        y_hard = (y_soft > 0.5).float()
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

class ModelC_Sparsity(nn.Module):
    def __init__(self, num_nodes, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.encoder = CausalDistributionEncoder(num_nodes, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.delta_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        
        self.dag_parent = nn.Linear(d_model, d_model)
        self.dag_child = nn.Linear(d_model, d_model)
        self.tau = 1.0 # Temperature

    def forward(self, base_samples, int_samples, target_row, int_mask, int_node_idx=None):
        x = self.encoder(base_samples, int_samples, target_row, int_mask)
        x = self.transformer(x)
        deltas = self.delta_head(x).squeeze(-1)
        
        p = self.dag_parent(x)
        c = self.dag_child(x)
        logits = torch.matmul(p, c.transpose(-2, -1))
        
        # Use Gumbel sampling for the DAG structure
        adj_sampled = gumbel_sigmoid(logits, tau=self.tau, hard=self.training)
        
        return deltas, logits, adj_sampled