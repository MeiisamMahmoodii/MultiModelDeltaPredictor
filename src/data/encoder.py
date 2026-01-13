import torch
import torch.nn as nn

class FourierEmbedding(nn.Module):
    def __init__(self, d_out, result_scale=1.0):
        super().__init__()
        self.d_out = d_out
        self.result_scale = result_scale
        # Frequencies: 2^0, 2^1, ... 2^7 (covers broad range)
        self.freqs = nn.Parameter(2.0 ** torch.arange(0, 8), requires_grad=False)
        self.proj = nn.Linear(len(self.freqs) * 2, d_out)
        
    def forward(self, x):
        # x: (..., 1)
        # Safety: Clip input to prevent numerical issues
        x = torch.clamp(x, -50, 50)
        # Periodic encoding
        x_proj = x * self.freqs.to(x.device) * torch.pi # (..., L)
        x_sin = torch.sin(x_proj)
        x_cos = torch.cos(x_proj)
        x_cat = torch.cat([x_sin, x_cos], dim=-1) # (..., 2L)
        result = self.proj(x_cat) * self.result_scale
        # Safety: Clip output to prevent overflow
        result = torch.clamp(result, -100, 100)
        return result

class HybridEmbedding(nn.Module):
    """
    Combines Linear, Fourier, and MLP embeddings to cover all physics types.
    """
    def __init__(self, d_model):
        super().__init__()
        
        # 1. Linear (Magnitude) - 1/4 capacity
        d_linear = d_model // 4
        self.linear_emb = nn.Linear(1, d_linear)
        
        # 2. Fourier (Periodicity) - 1/2 capacity
        d_fourier = d_model // 2
        self.fourier_emb = FourierEmbedding(d_fourier)
        
        # 3. MLP (Distortion/Sharpness) - 1/4 capacity
        d_mlp = d_model - d_linear - d_fourier
        self.mlp_emb = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, d_mlp)
        )
        
        # Mixer
        self.mixer = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (..., 1)
        # Safety: Clip input
        x = torch.clamp(x, -50, 50)
        
        l = self.linear_emb(x)
        f = self.fourier_emb(x)
        m = self.mlp_emb(x)
        
        # Safety: Clip embeddings to prevent overflow
        l = torch.clamp(l, -100, 100)
        f = torch.clamp(f, -100, 100)
        m = torch.clamp(m, -100, 100)
        
        cat = torch.cat([l, f, m], dim=-1) # (..., d_model)
        mixed = self.mixer(cat)
        # Safety: Clip mixed output
        mixed = torch.clamp(mixed, -100, 100)
        return self.norm(mixed)

class InterleavedEncoder(nn.Module):
    """
    Transforms Causal Data (Base, Mask, Value) into Interleaved Tokens.
    Format: [Feature_ID_0, Value_0, Feature_ID_1, Value_1, ...]
    
    Ablation Support: atomic_mode='additive' (Standard Transformer) vs 'interleaved' (Novel).
    """
    def __init__(self, num_vars, d_model, mode='interleaved'):
        super().__init__()
        self.num_vars = num_vars
        self.d_model = d_model
        self.mode = mode
        
        # Embeddings
        # 1. Feature ID Embedding: "Who am I?"
        self.var_id_emb = nn.Embedding(num_vars + 1, d_model)
        
        # 2. Value Embedding: "What is my value?"
        # Hybrid Strategy: Linear + Fourier + MLP
        self.value_emb = HybridEmbedding(d_model)
        
        # 3. Type Embedding: "Am I Observed, Intervened, or Masked?"
        # 0=Obs, 1=Int, 2=Masked
        self.type_emb = nn.Embedding(3, d_model)

    def forward(self, base_samples, int_samples, target_row, int_mask):
        """
        Args:
            base_samples: (B, N) - Current State (Refined in passes 2/3)
            int_samples: (B, N) - Intervention Values (Ground Truth for Intervened Node)
            target_row: (B, N) - Original Base (Unused if we use base_samples properly)
            int_mask: (B, N) - 1.0 if Intervened.
        """
        B, N = base_samples.shape
        device = base_samples.device
        
        # ISSUE 14: Model was ignoring input context and recurrence!
        # OLD: values = target_row.unsqueeze(-1)
        # NEW: Construct Effective Input State
        # For non-intervened nodes: Use base_samples (Current State)
        # For intervened nodes: Use int_samples (Forced Value)
        
        # Mix: val = base * (1 - mask) + int * mask
        # Note: int_mask might be boolean or float. Assume Float.
        mask = int_mask.unsqueeze(-1) # (B, N, 1)
        
        # Ensure shapes match
        if len(base_samples.shape) == 3:
             # If base_samples has context dim (B, Context, N), we flatten or take last?
             # CausalTransformer passes sequence? No, CausalTransformer passes (B, N).
             # Wait, CausalTransformer pass 2: refined_base might implement context later.
             # Current implementation: refined_base is (B, N).
             pass
             
        val_base = base_samples.unsqueeze(-1)
        val_int = int_samples.unsqueeze(-1)
        
        # Effective Values
        # Note: int_samples contains the target value for the intervened node.
        # base_samples contains the current belief of the state.
        # The tokens should represent the *State* we are querying about.
        # But for the Intervened Node, the value is FIXED by intervention.
        values = val_base * (1.0 - mask) + val_int * mask
        
        # 1. Feature Tokens
        ids = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1) # (B, N)
        f_emb = self.var_id_emb(ids) # (B, N, D)
        
        # 2. Value Tokens
        v_emb = self.value_emb(values) # (B, N, D)
        
        # Add Type Information
        t_ids = int_mask.long() # (B, N)
        t_emb = self.type_emb(t_ids) # (B, N, D)
        v_emb = v_emb + t_emb
        
        # 3. Combine
        if self.mode == 'interleaved':
            stacked = torch.stack([f_emb, v_emb], dim=2)
            tokens = stacked.flatten(1, 2)
        else:
            tokens = f_emb + v_emb
            
        return tokens
