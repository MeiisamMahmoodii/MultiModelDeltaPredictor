import torch
import torch.nn as nn
from src.data.CausalDistributionEncoder import CausalDistributionEncoder

def gumbel_sigmoid(logits, tau=1.0, hard=False):
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.sigmoid()

    if hard:
        y_hard = (y_soft > 0.5).float()
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

class ModelF_Unified(nn.Module):
    def __init__(self, num_nodes, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.encoder = CausalDistributionEncoder(num_nodes, d_model)
        
        # 1. Strong Backbone
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 2. Hyper-Experts (Combining Model B & E)
        # We have N experts, but their weights are modulated by the intervention context
        self.int_embedding = nn.Embedding(num_nodes, 32)
        
        # The HyperNet generates a modulation vector 'z' from the intervention
        self.hyper_net = nn.Sequential(
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, d_model) 
        )
        
        # Base Experts (like Model B)
        self.delta_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1)
            ) for _ in range(num_nodes)
        ])
        
        # 3. Sparse Structure Learning (Model C)
        self.dag_parent = nn.Linear(d_model, d_model)
        self.dag_child = nn.Linear(d_model, d_model)
        self.tau = 1.0

    def forward(self, base_samples, int_samples, target_row, int_mask, int_node_idx):
        # Encodings
        x = self.encoder(base_samples, int_samples, target_row, int_mask)
        x = self.transformer(x) # (B, N, d_model)
        
        # --- Hyper-Expert Delta Prediction ---
        # Get intervention context
        instr = self.int_embedding(int_node_idx) # (B, 32)
        modulation = self.hyper_net(instr).unsqueeze(1) # (B, 1, d_model)
        
        # Modulate the latent representation based on WHAT we intervened on
        # This tells the experts: "We poked Node 5, so adjust your logic accordingly"
        x_modulated = x * modulation
        
        deltas = []
        current_num_nodes = x.shape[1]
        for i in range(current_num_nodes):
            # Apply node-specific expert to the modulated input
            d_i = self.delta_experts[i](x_modulated[:, i, :])
            deltas.append(d_i)
        deltas = torch.cat(deltas, dim=-1)
        
        # --- Structural Discovery ---
        p = self.dag_parent(x)
        c = self.dag_child(x)
        logits = torch.matmul(p, c.transpose(-2, -1))
        
        # Use Gumbel for hard/sparse decisions
        adj_sampled = gumbel_sigmoid(logits, tau=self.tau, hard=self.training)
        
        return deltas, logits, adj_sampled
