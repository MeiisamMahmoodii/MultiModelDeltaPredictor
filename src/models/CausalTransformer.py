import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data.encoder import InterleavedEncoder

def gumbel_sigmoid(logits, tau=1.0, hard=False):
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight-through estimator
        y_hard = (y_soft > 0.5).float()
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # x: (..., d_model)
        norm = torch.mean(x.pow(2), dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm + self.eps)
        return self.scale * x_normed

class VectorizedSwiGLUResBlock(nn.Module):
    """
    Vectorized SwiGLU Block that runs 'num_experts' in parallel.
    Uses torch.einsum for efficient batched multiplication.
    """
    def __init__(self, d_model, num_experts, expansion_factor=8):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        dim_hidden = int(d_model * expansion_factor)
        
        self.norm = RMSNorm(d_model)
        
        # Weights: (Experts, In, Out)
        # Note: torch.nn.Parameter default init is not optimal for 3D, 
        # so we might need manual init or just rely on robust optimizers.
        scale = 1.0 / (d_model ** 0.5)
        self.w_gate = nn.Parameter(torch.randn(num_experts, d_model, dim_hidden) * scale)
        self.w_val = nn.Parameter(torch.randn(num_experts, d_model, dim_hidden) * scale)
        self.w_out = nn.Parameter(torch.randn(num_experts, dim_hidden, d_model) * scale)
        
    def forward(self, x):
        # x: (Tokens, Experts, d_model) - Input already expanded to experts
        # OR x: (Tokens, d_model) broadcast?
        # Let's assume input is (Tokens, Experts, d_model) so each expert has its own input stream
        
        residual = x
        x = self.norm(x) # RMSNorm applies to last dim
        
        # SwiGLU: (Tokens, Experts, d_model) @ (Experts, d_model, dim_hidden)
        # einsum 'ted, edh -> teh'
        gate = F.silu(torch.einsum('ted, edh -> teh', x, self.w_gate))
        val = torch.einsum('ted, edh -> teh', x, self.w_val)
        
        # Out: (Tokens, Experts, dim_hidden) @ (Experts, dim_hidden, d_model)
        # einsum 'teh, ehd -> ted'
        out = torch.einsum('teh, ehd -> ted', gate * val, self.w_out)
        
        return residual + out

class VectorizedDeepExpert(nn.Module):
    """
    A stack of Vectorized SwiGLU blocks.
    """
    def __init__(self, d_model, num_experts, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            VectorizedSwiGLUResBlock(d_model, num_experts) for _ in range(num_layers)
        ])
        
        # Final projection per expert
        # (Experts, d_model, 1)
        scale = 1.0 / (d_model ** 0.5)
        self.final_proj = nn.Parameter(torch.randn(num_experts, d_model, 1) * scale)

    def forward(self, x):
        # x: (Tokens, Experts, d_model)
        for layer in self.layers:
            x = layer(x)
        
        # Final: (Tokens, Experts, d_model) @ (Experts, d_model, 1) -> (Tokens, Experts, 1)
        return torch.einsum('ted, edc -> tec', x, self.final_proj).squeeze(-1) # (Tokens, Experts)

class MoELayer(nn.Module):
    """
    Vectorized Mixture of Experts Layer.
    """
    def __init__(self, d_model, num_experts=8, num_layers_per_expert=4):
        super().__init__()
        self.num_experts = num_experts
        
        # 1. Vectorized Experts
        self.experts = VectorizedDeepExpert(d_model, num_experts, num_layers_per_expert)
        
        # 2. Router
        self.router = nn.Linear(d_model, num_experts)
        
    def forward(self, x):
        """
        x: (Batch, Num_Active, d_model)
        """
        batch_size, num_active, d_model = x.shape
        
        # Flatten: (Total_Tokens, d_model)
        x_flat = x.flatten(0, 1)
        total_tokens = x_flat.size(0)
        
        # 1. Router
        logits = self.router(x_flat) # (Total, Experts)
        weights = F.softmax(logits, dim=-1) # (Total, Experts)
        
        # 2. Vectorized Execution
        # We broadcast x to all experts: (Total, Experts, d_model)
        x_expanded = x_flat.unsqueeze(1).expand(-1, self.num_experts, -1)
        
        # Run all experts in parallel
        expert_outputs = self.experts(x_expanded) # (Total, Experts)
        
        # 3. Weighted Sum
        output = (expert_outputs * weights).sum(dim=1) # (Total, )
        
        # Reshape back to batch
        return output.view(batch_size, num_active, 1)

class CausalTransformer(nn.Module):
    """
    Unified Causal Transformer (Phase 3 "Physics-Native")
    Features:
    - HybridValueEncoding (Fourier+MLP+Linear)
    - Vectorized MoE
    - Universal Multi-Node Interventions
    """
    def __init__(self, num_nodes, d_model=256, nhead=8, num_layers=12):
        super().__init__()
        self.num_nodes = num_nodes
        
        # 1. Interleaved Encoding (With Hybrid Value Embeddings)
        self.encoder = InterleavedEncoder(num_nodes, d_model)
        
        # 2. Strong Backbone
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Universal MoE Layer (Vectorized)
        self.moe = MoELayer(d_model, num_experts=8, num_layers_per_expert=4)
        
        # 4. Sparse Structure Learning
        self.dag_parent = nn.Linear(d_model, d_model)
        self.dag_child = nn.Linear(d_model, d_model)
        self.tau = 1.0

    def forward(self, base_samples, int_samples, target_row, int_mask, int_node_idx=None):
        # Note: int_node_idx is DEPRECATED and ignored (kept for API compatibility)
        
        # Encodings: (B, 2N, D)
        # InterleavedEncoder handles the 'TypeEmbedding' (Observed vs Intervened)
        # using 'int_mask'.
        x = self.encoder(base_samples, int_samples, target_row, int_mask)
        x = self.transformer(x) 
        
        feature_tokens = x[:, 0::2, :] # (B, N, D)
        value_tokens = x[:, 1::2, :]   # (B, N, D)
        
        # --- Value Delta Prediction ---
        # No HyperNet. The 'value_tokens' already contain context from 'int_mask' 
        # via the Transformer's self-attention mixing.
        # We pass directly to MoE.
        deltas = self.moe(value_tokens)
        
        # --- Structural Discovery ---
        p = self.dag_parent(feature_tokens)
        c = self.dag_child(feature_tokens)
        logits = torch.matmul(p, c.transpose(-2, -1))
        
        adj_sampled = gumbel_sigmoid(logits, tau=self.tau, hard=self.training)
        
        return deltas, logits, adj_sampled

    def anneal_temperature(self, epoch, total_epochs):
        self.tau = max(0.1, 1.0 - (epoch / total_epochs))
        return self.tau
