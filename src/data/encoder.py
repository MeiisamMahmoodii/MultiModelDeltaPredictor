import torch
import torch.nn as nn

# Removed FourierEmbedding and HybridEmbedding as they are replaced by simple Linear
# per the "Scientist-Engineer" prompt requirements.

class PeriodicLinearEmbedding(nn.Module):
    def __init__(self, d_model, sigma=1.0):
        super().__init__()
        self.d_model = d_model
        self.sigma = sigma
        
        # 1. Linear Branch (Magnitude)
        # We allocate half the dimension to simple linear projection
        self.d_linear = d_model // 2
        self.linear = nn.Linear(1, self.d_linear)
        
        # 2. Periodic Branch (Physics/Details)
        # We allocate the other half to Sin/Cos pairs
        self.d_periodic = d_model - self.d_linear
        # Frequency weights are TRAINABLE parameters initialized from Normal(0, sigma)
        # Div 2 because we generate sin/cos pairs for each frequency
        self.w_freq = nn.Parameter(torch.randn(1, self.d_periodic // 2) * sigma)
        self.p_proj = nn.Linear(self.d_periodic, self.d_periodic)
        
    def forward(self, x):
        # x: (Batch, N, 1)
        
        # Linear: Captures "x is 100 vs 0"
        v_lin = self.linear(x) 
        
        # Periodic: Captures "x is part of a wave"
        # x @ w_freq -> (Batch, N, d_periodic//2)
        # Broadcast x (B, N, 1) against w_freq (1, d_p//2)
        # We want inner prod? No, x is scalar value per node.
        # x is (B,N,1). w_freq is (1, F). Result (B,N,F).
        x_freq = 2 * torch.pi * x * self.w_freq 
        
        v_per = torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1)
        v_per = self.p_proj(v_per)
        
        return torch.cat([v_lin, v_per], dim=-1)

class AdditiveEncoder(nn.Module):
    """
    Encodes Causal Data (ID, Value, Type) using an Additive strategy:
    Embedding = Embed(ID) + PLR(Value) + Embed(Type)
    
    This replaces the memory-heavy InterleavedEncoder.
    """
    def __init__(self, num_vars, d_model):
        super().__init__()
        self.num_vars = num_vars
        self.d_model = d_model
        
        # 1. Feature ID Embedding: "Who am I?"
        self.var_id_emb = nn.Embedding(num_vars, d_model)
        
        # 2. Value Embedding: "What is my value?"
        # Using Trainable Periodic Linear Embeddings (PLR)
        self.value_emb = PeriodicLinearEmbedding(d_model)
        
        # 3. Type Embedding: "Am I Observed, Intervened, or Masked?"
        # 0=Obs, 1=Int, 2=Masked
        self.type_emb = nn.Embedding(3, d_model)

    def forward(self, base_samples, int_samples, target_row, int_mask):
        """
        Args:
            base_samples: (B, N) - Current State
            int_samples: (B, N) - Intervention Values
            target_row: (B, N) - (Unused mostly, but kept for signature compat)
            int_mask: (B, N) - 1.0 if Intervened.
        """
        B, N = base_samples.shape
        device = base_samples.device
        
        # Construct Effective Input Values
        # For non-intervened nodes: Use base_samples
        # For intervened nodes: Use int_samples
        mask = int_mask.unsqueeze(-1) # (B, N, 1)
        val_base = base_samples.unsqueeze(-1)
        val_int = int_samples.unsqueeze(-1)
        
        values = val_base * (1.0 - mask) + val_int * mask # (B, N, 1)
        
        # 1. ID Embedding
        ids = torch.arange(N, device=device).unsqueeze(0).expand(B, N) # (B, N)
        id_emb = self.var_id_emb(ids) # (B, N, D)
        
        # 2. Value Embedding (PLR)
        val_emb = self.value_emb(values) # (B, N, D)
        
        # 3. Type Embedding
        # int_mask is likely float, cast to long. 
        # CAUTION: if int_mask has values like 2.0 (masked in MCM), we need to handle that.
        # Assuming int_mask contains 0, 1, 2.
        type_ids = int_mask.long()
        type_emb = self.type_emb(type_ids) # (B, N, D)
        
        # Additive combination
        x = id_emb + val_emb + type_emb
        
        return x, ids # Return embedding and pos_ids (just ids here)
