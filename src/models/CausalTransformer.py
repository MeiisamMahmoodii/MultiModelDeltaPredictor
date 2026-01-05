import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from src.data.encoder import InterleavedEncoder
from src.models.rope import RotaryEmbedding, apply_rotary_pos_emb

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
    Refactored for Hard Gumbel Routing (Scenario 7).
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
        
        # 1. Router (Hard Gumbel Switch)
        logits = self.router(x_flat) # (Total, Experts)
        # Use Gumbel Softmax (Hard=True) for crisp expert selection
        # During training, this uses straight-through estimator
        weights = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1) # (Total, Experts)
        
        # 2. Vectorized Execution
        # We broadcast x to all experts: (Total, Experts, d_model)
        x_expanded = x_flat.unsqueeze(1).expand(-1, self.num_experts, -1)
        
        # Run all experts in parallel
        expert_outputs = self.experts(x_expanded) # (Total, Experts)
        
        # 3. Weighted Sum (In Hard mode, this selects exactly one expert per token)
        output = (expert_outputs * weights).sum(dim=1) # (Total, )
        
        # Reshape back to batch: (Batch, Num_Active)
        return output.view(batch_size, num_active)

class RoPEAttentionLayer(nn.Module):
    """
    Custom Transformer Layer with Rotary Positional Embeddings.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation Note: PyTorch MHA doesn't easily expose Q/K for RoPE.
        # So we use a simplified manual Attention or accept that we rotate X before MHA?
        # Rotating X before MHA corresponds to xPos-like behavior but isn't strict RoPE.
        # Strict RoPE requires rotating Q and K *after* linear projection but *before* dot product.
        # To do this correctly without rewriting MHA, we use a manual attention implementation OR
        # since PyTorch 2.1 SDPA allows it, but for compatibility here we might write a simple
        # FlashAttention-like wrapper if we want speed.
        #
        # DECISION: For stability and simplicity in "Physics First" phase, 
        # let's assume we implement a basic custom Attention block.
        
        self.head_dim = d_model // nhead
        self.nhead = nhead
        self.d_model = d_model
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Feed Forward
        self.norm1 = nn.LayerNorm(d_model) # (Or RMSNorm if preferred)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rotary_emb=None):
        # x: (Batch, Seq, D)
        B, S, D = x.shape
        
        # Residual 1
        residual = x
        x = self.norm1(x)
        
        # QKV
        q = self.w_q(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2) # (B, H, S, Hd)
        k = self.w_k(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if rotary_emb is not None:
            cos, sin = rotary_emb(v, seq_len=S) # Get cos/sin for this sequence length
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            
        # Attention
        # scaled_dot_product_attention handles masking/dropout efficiently
        # We don't have causal mask logic passed here yet, generally is_causal=False for "All-to-All"
        # unless we were masking for DAG (Scenario 2). For now, All-to-All.
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)
        
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.w_o(out)
        x = residual + self.dropout(out)
        
        # Residual 2
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.dropout(x)
        
        return x

class RoPETransformer(nn.Module):
    """
    Stack of RoPE Layers. Replacement for nn.TransformerEncoder.
    """
    def __init__(self, d_model, nhead, num_layers, max_len=2048):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPEAttentionLayer(d_model, nhead) for _ in range(num_layers)
        ])
        self.rope = RotaryEmbedding(d_model // nhead, max_position_embeddings=max_len)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x, self.rope)
        return x

class CausalTransformer(nn.Module):
    """
    Unified Causal Transformer (Phase 4 "Physics-First")
    Features:
    - RoPE (Relative Positions)
    - Vectorized MoE (Hard Gumbel)
    - Recurrent Refinement (3-step)
    - NO DAG HEAD (Decoupled)
    """
    def __init__(self, num_nodes, d_model=256, nhead=8, num_layers=12, grad_checkpoint=False, 
                 ablation_dense=False, ablation_no_interleaved=False, ablation_no_dag=False, ablation_no_physics=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.grad_checkpoint = grad_checkpoint
        
        self.ablation_no_interleaved = ablation_no_interleaved
        self.ablation_no_dag = ablation_no_dag
        self.ablation_no_physics = ablation_no_physics
        
        # 1. Interleaved Encoding (or Additive for Ablation)
        mode = 'additive' if ablation_no_interleaved else 'interleaved'
        self.encoder = InterleavedEncoder(num_nodes, d_model, mode=mode)
        
        # 2. RoPE Backbone (Replaces standard TransformerEncoder)
        # If additive (no interleaved), seq_len is N (plus margin).
        # If interleaved, seq_len is 2N.
        factor = 1 if ablation_no_interleaved else 2
        self.transformer = RoPETransformer(d_model, nhead, num_layers, max_len=num_nodes*factor + 10)
        
        # 3. Universal MoE Layer (Hard Gumbel)
        # Ablation: Dense = 1 Expert (Trivial Routing)
        num_experts = 1 if ablation_dense else 8
        self.moe = MoELayer(d_model, num_experts=num_experts, num_layers_per_expert=4)
        
        # 4. Masked Causal Modeling Head (Pre-training)
        self.mcm_head = nn.Linear(d_model, 1)

        # 5. DAG Structural Head (Phase 5: Re-enabled)
        # We want to predict Adjacency Matrix A_ij from Pairs of Tokens (i, j)
        # But for O(N^2) efficiency, we can project to Key/Query space and do dot product?
        # Or simplistic: Predict 'Parent' logits for each node.
        # Strategy: Each node emits a "Parent Query". All nodes emit "Parent Keys".
        # This is essentially one attention block trained to find parents.
        self.dag_query = nn.Linear(d_model, d_model)
        self.dag_key = nn.Linear(d_model, d_model)
        self.dag_scale = d_model ** -0.5

    def forward(self, base_samples, int_samples, target_row, int_mask, int_node_idx=None, mcm_mask=None):
        """
        Input:
            mcm_mask: (B, N) or None. If present, 1.0 means this token is masked.
        """
        # Pass 1: Initial Guess
        if self.grad_checkpoint and self.training:
            # Inputs to pass 1 don't require grad, so use_reentrant=False is mandatory for checkpoint to work
            deltas_1, mcm_out, logits_1 = checkpoint(self._forward_pass, base_samples, int_samples, target_row, int_mask, mcm_mask, use_reentrant=False)
        else:
            deltas_1, mcm_out, logits_1 = self._forward_pass(base_samples, int_samples, target_row, int_mask, mcm_mask)
        
        if mcm_mask is not None:
             B, N = base_samples.shape
             dummy_logits = torch.zeros(B, N, N, device=base_samples.device)
             dummy_adj = torch.zeros(B, N, N, device=base_samples.device)
             return deltas_1, dummy_logits, dummy_adj, mcm_out
        
        
        # Pass 2: Refinement (Recurrent)
        # Broadcast deltas (B, N) to (B, Context, N) if base_samples has context dim
        if len(base_samples.shape) == 3:
             refined_base = base_samples + deltas_1.unsqueeze(1)
        else:
             refined_base = base_samples + deltas_1
             
        if self.grad_checkpoint and self.training:
            deltas_2, _, _ = checkpoint(self._forward_pass, refined_base, int_samples, target_row, int_mask, None, use_reentrant=False)
        else:
            deltas_2, _, _ = self._forward_pass(refined_base, int_samples, target_row, int_mask, None)
        
        
        # Pass 3: Final Polish
        if len(base_samples.shape) == 3:
             refined_base_2 = base_samples + deltas_2.unsqueeze(1)
        else:
             refined_base_2 = base_samples + deltas_2
             
        if self.grad_checkpoint and self.training:
             deltas_final, _, logits_final = checkpoint(self._forward_pass, refined_base_2, int_samples, target_row, int_mask, None, use_reentrant=False)
        else:
             deltas_final, _, logits_final = self._forward_pass(refined_base_2, int_samples, target_row, int_mask, None)
        
        
        # Returning Dummy Logits/Adj for API compatibility with main.py metrics
        if len(base_samples.shape) == 3:
            B, S, N = base_samples.shape
        else:
            B, N = base_samples.shape
            
        dummy_adj = torch.zeros(B, N, N, device=base_samples.device)
        
        # Phase 5: Return REAL logits from final pass
        # We need to call _forward_pass one last time BUT ask it to compute structure
        # actually _forward_pass returns deltas, mcm_out, (and now) logits
        # Let's adjust _forward_pass to return logits too.
        
        return deltas_final, logits_final, dummy_adj, None

    def _forward_pass(self, base_samples, int_samples, target_row, int_mask, mcm_mask):
        # Prepare inputs for Encoder (Handle masking)
        enc_int_mask = int_mask
        enc_target = target_row
        
        if mcm_mask is not None:
            # Clone to avoid in-place modification of inputs which might be used later?
            # Creating combined mask: 2 if Masked, else int_mask
            enc_int_mask = int_mask.clone()
            enc_int_mask[mcm_mask.bool()] = 2.0 # Type 2 = Masked
            
            # Zero out the values
            enc_target = target_row.clone()
            enc_target[mcm_mask.bool()] = 0.0
            
        # Shared Forward Logic
        x = self.encoder(base_samples, int_samples, enc_target, enc_int_mask)
        x = self.transformer(x)
        
        # Extract Value Tokens (embedding of the variable values)
        if self.ablation_no_interleaved:
            # Mode 'additive': All tokens are value tokens
            value_tokens = x
        else:
            # Mode 'interleaved': [ID, Value, ID, Value...]
            value_tokens = x[:, 1::2, :]   # (B, N, D)
            
        # Physics Head (MoE)
        if self.ablation_no_physics:
             # Return zeros
             # deltas shape: (B, N)
             deltas = torch.zeros(value_tokens.shape[0], value_tokens.shape[1], device=value_tokens.device)
        else:
             deltas = self.moe(value_tokens)
        
        mcm_out = None
        if mcm_mask is not None:
            mcm_out = self.mcm_head(value_tokens).squeeze(-1) # (B, N)
            
        mcm_out = None
        if mcm_mask is not None:
            mcm_out = self.mcm_head(value_tokens).squeeze(-1) # (B, N)
            
        # DAG Prediction (Phase 5)
        logits = torch.zeros(value_tokens.shape[0], value_tokens.shape[1], value_tokens.shape[1], device=value_tokens.device)
        if not self.ablation_no_dag:
            # Query: "Who are my parents?"
            Q = self.dag_query(value_tokens) # (B, N, D)
            # Key: "I am a potential parent"
            K = self.dag_key(value_tokens)   # (B, N, D)
            
            logits = torch.matmul(Q, K.transpose(-2, -1)) * self.dag_scale
        
        return deltas, mcm_out, logits

    def anneal_temperature(self, epoch, total_epochs):
        # No tau needed for Gumbel (Hard=True)
        return 1.0

