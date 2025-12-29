import torch
import torch.nn as nn
from src.data.encoder import InterleavedEncoder

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

class CausalTransformer(nn.Module):
    """
    Unified Causal Transformer (formerly Model F)
    Features:
    - Interleaved Tokenization (Feature, Value)
    - Hyper-Interventional Experts
    - Sparse Structure Learning (Gumbel)
    """
    def __init__(self, num_nodes, d_model=256, nhead=8, num_layers=12):
        super().__init__()
        self.num_nodes = num_nodes
        
        # 1. Interleaved Encoding
        self.encoder = InterleavedEncoder(num_nodes, d_model)
        
        # 2. Strong Backbone
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Hyper-Experts
        # Intervention Embedding
        self.int_embedding = nn.Embedding(num_nodes, 32)
        
        # HyperNet: Intervention -> Latent Modulation
        self.hyper_net = nn.Sequential(
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, d_model) 
        )
        
        # Node-Specific Experts (Process Feature Tokens)
        self.delta_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1) # Predicts Delta
            ) for _ in range(num_nodes)
        ])
        
        # 4. Sparse Structure Learning
        # Estimates Adjacency matrix from Feature Tokens
        self.dag_parent = nn.Linear(d_model, d_model)
        self.dag_child = nn.Linear(d_model, d_model)
        self.tau = 1.0

    def forward(self, base_samples, int_samples, target_row, int_mask, int_node_idx):
        # Encodings: (B, 2N, D)
        # Sequence: [F0, V0, F1, V1, ...]
        x = self.encoder(base_samples, int_samples, target_row, int_mask)
        x = self.transformer(x) 
        
        # Separate Feature and Value representations
        # Feature Tokens are at indices 0, 2, 4... -> represented variable identity + structure
        # Value Tokens are at indices 1, 3, 5... -> represented current state
        
        # For Experts, we want to combine both. 
        # But our Experts are Node-Specific. 
        # Let's extract the FEATURE tokens for structure and VALUE tokens for state?
        # Actually Model F design was: Modulate the representation, then Expert.
        
        # Strategy: Use the FEATURE tokens (even indices) for Structure (Adjacency)
        # Use the VALUE tokens (odd indices) for Delta Prediction (State)
        
        feature_tokens = x[:, 0::2, :] # (B, N, D)
        value_tokens = x[:, 1::2, :]   # (B, N, D)
        
        # --- Hyper-Expert Delta Prediction ---
        # Get intervention context
        instr = self.int_embedding(int_node_idx) # (B, 32)
        modulation = self.hyper_net(instr).unsqueeze(1) # (B, 1, d_model)
        
        # Modulate Value Tokens ("The state has changed rules")
        x_modulated = value_tokens * modulation
        
        deltas = []
        # Dynamic loop for current batch size (might be < self.num_nodes in Curriculum)
        current_num_nodes = value_tokens.shape[1]
        
        for i in range(current_num_nodes):
            # Apply node-specific expert
            d_i = self.delta_experts[i](x_modulated[:, i, :])
            deltas.append(d_i)
        deltas = torch.cat(deltas, dim=-1) # (B, N)
        
        # --- Structural Discovery ---
        # Use Feature Tokens (Structural Identity)
        p = self.dag_parent(feature_tokens)
        c = self.dag_child(feature_tokens)
        logits = torch.matmul(p, c.transpose(-2, -1))
        
        # Use Gumbel for hard/sparse decisions
        adj_sampled = gumbel_sigmoid(logits, tau=self.tau, hard=self.training)
        
        return deltas, logits, adj_sampled

    def anneal_temperature(self, epoch, total_epochs):
        # Simple annealing schedule
        self.tau = max(0.1, 1.0 - (epoch / total_epochs))
        return self.tau
