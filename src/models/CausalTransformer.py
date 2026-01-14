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

class SimpleMoELayer(nn.Module):
    """
    Simple, reliable Mixture of Experts layer.
    Each expert is a 2-layer MLP. Router uses Gumbel-Softmax for discrete routing.
    """
    def __init__(self, d_model, num_experts=8, expansion_factor=4):
        super().__init__()
        self.num_experts = num_experts
        
        # Router: (d_model) -> (num_experts)
        self.router = nn.Linear(d_model, num_experts)
        
        # Experts: each is a simple FFN
        dim_hidden = int(d_model * expansion_factor)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_hidden),
                nn.GELU(),
                nn.Linear(dim_hidden, d_model)
            )
            for _ in range(num_experts)
        ])
        
        # Usage tracking
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))
        
    def forward(self, x):
        """
        x: (B, N, d_model)
        Returns: (B, N, d_model), aux_loss
        """
        B, N, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (B*N, d_model)
        
        # Route tokens to experts
        logits = self.router(x_flat)  # (B*N, num_experts)
        weights = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)  # (B*N, num_experts)
        
        # Compute load balancing loss
        probs = F.softmax(logits, dim=-1)
        
        # 1. Local Statistics
        local_prob_sum = probs.sum(dim=0) # (Experts,)
        local_count = torch.tensor(probs.size(0), device=probs.device, dtype=probs.dtype)
        
        # 2. Global Synchronization (DDP)
        if torch.distributed.is_initialized():
            # Sync sum of probabilities
            local_prob_sum = differentiable_all_reduce_sum(local_prob_sum)
            # Sync total token count
            local_count = differentiable_all_reduce_sum(local_count)
            
        # 3. Compute Global Importance
        global_count = torch.clamp(local_count, min=1.0)
        importance = local_prob_sum / global_count # (Experts,)
        
        target = 1.0 / self.num_experts
        aux_loss = torch.mean((importance - target) ** 2)
        
        # Track usage
        if self.training:
            with torch.no_grad():
                usage = weights.sum(dim=0)
                self.expert_counts += usage
                self.total_tokens += x_flat.size(0)
        
        # Route through experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_outputs.append(expert(x_flat))  # (B*N, d_model)
        
        # Stack: (num_experts, B*N, d_model)
        expert_outputs = torch.stack(expert_outputs, dim=0)
        
        # Select outputs via routing weights
        # weights: (B*N, num_experts) -> transpose to (num_experts, B*N)
        # expert_outputs: (num_experts, B*N, d_model)
        weights_t = weights.t()  # (num_experts, B*N)
        
        # Weighted sum: sum over experts dimension
        # (num_experts, B*N, 1) * (num_experts, B*N, d_model) -> (B*N, d_model)
        output = torch.einsum('ej,ejd->jd', weights_t, expert_outputs)
        
        return output.view(B, N, d_model), aux_loss
    
    def reset_metrics(self):
        self.expert_counts.zero_()
        self.total_tokens.zero_()
    
    def get_expert_metrics(self):
        counts = self.expert_counts.float()
        total = self.total_tokens.item()
        if total == 0:
            return {"entropy": 0.0, "gini": 0.0, "counts": [0.0] * self.num_experts}
        
        probs = counts / total
        # Entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        # Gini
        sorted_probs = torch.sort(probs)[0]
        n = len(sorted_probs)
        gini = (2.0 * torch.sum((n + 1 - torch.arange(1, n+1).to(probs.device)) * sorted_probs) / (n * torch.sum(probs))) - (n + 1) / n
        gini = gini.item()
        
        return {"entropy": entropy, "gini": gini, "counts": counts.cpu().numpy().tolist()}


class MoELayer(nn.Module):
    """
    Vectorized Mixture of Experts Layer.
    Refactored for Hard Gumbel Routing (Scenario 7).
    """
    def __init__(self, d_model, num_experts=8, num_layers_per_expert=4, router_tau=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.tau = router_tau
        
        # 1. Vectorized Experts
        self.experts = VectorizedDeepExpert(d_model, num_experts, num_layers_per_expert)
        
        # 2. Router
        self.router = nn.Linear(d_model, num_experts)
        
        # 3. Usage Monitoring (Persistent Buffers)
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))
        
    def forward(self, x):
        """
        x: (Batch, Num_Active, d_model)
        Returns: output, aux_loss
        """
        batch_size, num_active, d_model = x.shape
        
        # Flatten: (Total_Tokens, d_model)
        x_flat = x.flatten(0, 1)
        
        # 1. Router (Hard Gumbel Switch)
        logits = self.router(x_flat) # (Total, Experts)
        # Use Gumbel Softmax (Hard=True) for crisp expert selection
        weights = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1) # (Total, Experts)
        
        # Aux Loss: Load Balancing
        probs = F.softmax(logits, dim=-1) # (Total, Experts)
        
        # 1. Local Statistics
        local_prob_sum = probs.sum(dim=0) # (Experts,)
        local_count = torch.tensor(probs.size(0), device=probs.device, dtype=probs.dtype)
        
        # 2. Global Synchronization (DDP)
        is_ddp = torch.distributed.is_initialized()
        print(f"DEBUG: torch.distributed.is_initialized() = {is_ddp}")
        if is_ddp:
            # Sync sum of probabilities
            torch.distributed.all_reduce(local_prob_sum, op=torch.distributed.ReduceOp.SUM)
            # Sync total token count
            torch.distributed.all_reduce(local_count, op=torch.distributed.ReduceOp.SUM)
            
        # 3. Compute Global Importance
        # Avoid division by zero
        global_count = torch.clamp(local_count, min=1.0)
        importance = local_prob_sum / global_count # (Experts,)
        
        # Target: Importance should be uniform (1 / N)
        target = 1.0 / self.num_experts
        aux_loss = torch.mean((importance - target)**2) # MSE from Uniform Distribution
        
        # Usage Tracking
        if self.training:
            with torch.no_grad():
                usage = weights.sum(dim=0).detach()
                self.expert_counts += usage
                self.total_tokens += weights.size(0)
        
        # 2. Vectorized Execution
        # Broadcast x to all experts: (Total, Experts, d_model)
        x_expanded = x_flat.unsqueeze(1).expand(-1, self.num_experts, -1)
        expert_outputs = self.experts(x_expanded)  # (Total, Experts)
        
        # 3. Reconstruction: expert_outputs are scalars, weights are one-hot
        # Approach: treat expert outputs as scaling factors for the input
        # Scale input by expert logits and sum across experts
        # weights shape: (Total, Experts), expert_outputs shape: (Total, Experts)
        scaled = expert_outputs * weights  # (Total, Experts)
        
        # Aggregate: sum across experts and multiply with input
        # This is a scaling + gating mechanism
        scale = scaled.sum(dim=1, keepdim=True)  # (Total, 1)
        output = x_flat * scale  # (Total, d_model)
        
        return output.view(batch_size, num_active), aux_loss

    def reset_metrics(self):
        """Resets expert usage counters for new epoch"""
        self.expert_counts.zero_()
        self.total_tokens.zero_()

    def get_expert_metrics(self):
        """
        Computes entropy and gini coefficient of expert usage.
        Returns dict.
        """
        if self.total_tokens == 0:
            return {"entropy": 0.0, "gini": 0.0}
            
        probs = self.expert_counts / self.total_tokens
        
        # Entropy: -sum(p * log(p)), with safety guards
        # Clamp probabilities to avoid log(0)
        probs_safe = torch.clamp(probs, min=1e-10, max=1.0)
        entropy = -torch.sum(probs_safe * torch.log(probs_safe))
        
        sorted_probs, _ = torch.sort(probs)
        n = self.num_experts
        index = torch.arange(1, n + 1, device=probs.device, dtype=probs.dtype)
        gini = (2.0 * torch.sum(index * sorted_probs) / (n * torch.sum(sorted_probs) + 1e-10)) - (n + 1.0) / n
        
        entropy_val = entropy.item()
        gini_val = gini.item()
        
        # Safety checks
        entropy_val = 0.0 if (entropy_val != entropy_val) or (entropy_val > 1e6) else entropy_val
        gini_val = 0.0 if (gini_val != gini_val) or (gini_val > 1e6) else gini_val
        
        return {
            "entropy": entropy_val, 
            "gini": gini_val,
            "counts": self.expert_counts.cpu().numpy().tolist()
        }

class LearnedCausalMask(nn.Module):
    """
    Learn which attention connections respect causality using predicted adjacency.
    """
    def __init__(self, num_nodes):
        super().__init__()
        # Learn scaling parameter for the mask (how strong should the causal bias be?)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, adj_logits):
        """
        adj_logits: (B, N, N) where A[i, j] high means i -> j
        Attention: Target attends to Source.
        If i -> j, then j depends on i. So j should attend to i.
        Attn[Target=j, Source=i] should be high.
        So we need Transpose(adj_logits).
        """
        # (B, N, N) -> (B, N, N)
        # Transpose so that Attn[j, i] corresponds to Edge i->j
        causal_bias = adj_logits.transpose(1, 2) 
        
        # Scale and Bias
        # We return logits to be added to attention scores
        return (causal_bias * self.scale) + self.bias

class RoPEAttentionLayer(nn.Module):
    """
    Custom Transformer Layer with Rotary Positional Embeddings and MoE.
    MoE replaces the standard FFN for increased capacity and specialization.
    """
    def __init__(self, d_model, nhead, num_experts=8, dropout=0.1, router_tau=1.0):
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
        
        # MoE replaces standard FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = SimpleMoELayer(d_model, num_experts=num_experts, expansion_factor=4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rotary_emb=None, pos_ids=None, attn_mask=None):
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
            # Pass pos_ids (Variable IDs) if available for set-based equivariance
            cos, sin = rotary_emb(q, seq_len=S, position_ids=pos_ids) 
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            
        # Attention
        # scaled_dot_product_attention handles masking/dropout efficiently
        # attn_mask: (B, S, S) or broadcastable
        # Ensure mask is broadcasted to heads if needed
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.1)
        
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.w_o(out)
        x = residual + self.dropout(out)
        
        # Residual 2: MoE (replaces FFN)
        residual = x
        x = self.norm2(x)
        x_moe, aux_loss = self.moe(x)
        x = residual + self.dropout(x_moe)
        
        return x, aux_loss

class RoPETransformer(nn.Module):
    """
    Stack of RoPE Layers with MoE in each layer.
    """
    def __init__(self, d_model, nhead, num_layers, num_experts=8, max_len=2048, router_tau=1.0):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPEAttentionLayer(d_model, nhead, num_experts=num_experts, router_tau=router_tau) 
            for _ in range(num_layers)
        ])
        self.rope = RotaryEmbedding(d_model // nhead, max_position_embeddings=max_len)
        
    def forward(self, x, pos_ids=None, attn_mask=None):
        total_aux = 0.0
        for layer in self.layers:
            x, aux_loss = layer(x, self.rope, pos_ids=pos_ids, attn_mask=attn_mask)
            total_aux += aux_loss
        return x, total_aux

class CausalTransformer(nn.Module):
    """
    Unified Causal Transformer (Phase 4 "Physics-First" Architecture).

    A specialized Transformer designed to predict the *consequences* of interventions (deltas)
    rather than just the graph structure.

    Architectural Pillars:
    1.  **Rotary Positional Embeddings (RoPE)**:
        - Encodes variable identity ($X_1$ vs $X_2$) relative to others, preserving permutation equivariance where appropriate.
    2.  **Hard-Gumbel Mixture of Experts (MoE)**:
        - Replaces the standard FFN with 8 "Expert" MLPs.
        - Using Hard Gumbel Softmax ensures discrete routing (each token goes to specific experts),
          allowing experts to specialize in different functional forms (Linear, Polynomial, Step, etc.).
    3.  **Learned Causal Mask**:
        - A dedicated head predicts a soft adjacency matrix ($A$) from the input context.
        - This matrix biases the Self-Attention mechanism, forcing the model to attend to "Parents" when predicting "Children".
    4.  **Iterative Structure Refinement (3-Pass)**:
        - **Pass 1**: Rapid guess of structure and deltas.
        - **Pass 2**: Refine the Causal Mask based on Pass 1's structure.
        - **Pass 3**: Final Delta prediction using the high-fidelity mask.
        - This allows gradients from the Physics Loss (MAE) to backpropagate and refine the Structure Head.
    """
    """
    Unified Causal Transformer (Phase 4 "Physics-First")
    Features:
    - RoPE (Relative Positions)
    - Vectorized MoE (Hard Gumbel)
    - Recurrent Refinement (3-step)
    - NO DAG HEAD (Decoupled)
    """
    def __init__(self, num_nodes, d_model=256, nhead=8, num_layers=12, grad_checkpoint=False, 
                 ablation_dense=False, ablation_no_interleaved=False, ablation_no_dag=False, ablation_no_physics=False,
                 router_tau=1.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.grad_checkpoint = grad_checkpoint
        
        self.ablation_no_interleaved = ablation_no_interleaved
        self.ablation_no_dag = ablation_no_dag
        self.ablation_no_physics = ablation_no_physics
        
        # 1. Interleaved Encoding (or Additive for Ablation)
        mode = 'additive' if ablation_no_interleaved else 'interleaved'
        self.encoder = InterleavedEncoder(num_nodes, d_model, mode=mode)
        
        # 2. RoPE Backbone with MoE in each layer
        # If additive (no interleaved), seq_len is N (plus margin).
        # If interleaved, seq_len is 2N.
        factor = 1 if ablation_no_interleaved else 2
        # ISSUE 16: Position Embedding Size Mismatch
        # Graphs can vary or exceed initial estimate. Use large safety buffer (4096).
        # RoPE adjusts dynamically, but large init prevents reallocation jitter.
        num_experts = 1 if ablation_dense else 8
        self.transformer = RoPETransformer(
            d_model, nhead, num_layers, 
            num_experts=num_experts, 
            max_len=4096, 
            router_tau=router_tau
        )
        
        # 3. Physics Head: Simple projection from token embeddings to delta predictions
        self.physics_head = nn.Linear(d_model, 1)
        # Phase 4: Uncertainty Head for I-NLL (Generative Metric)
        # Predicts log(sigma^2) or log(sigma)
        self.uncertainty_head = nn.Linear(d_model, 1)
        
        # 4. Learned Causal Mask (Phase 3 Hotfix: Topology-Aware Attention)
        # "Novel Solution: Learned Causal Masking"
        # We learn to bias attention based on predicted adjacency.
        self.causal_mask_net = LearnedCausalMask(num_nodes)
        


        # 5. DAG Structural Head (Phase 5: Re-enabled)
        # We want to predict Adjacency Matrix A_ij from Pairs of Tokens (i, j)
        # But for O(N^2) efficiency, we can project to Key/Query space and do dot product?
        # Or simplistic: Predict 'Parent' logits for each node.
        # Strategy: Each node emits a "Parent Query". All nodes emit "Parent Keys".
        # This is essentially one attention block trained to find parents.
        self.dag_query = nn.Linear(d_model, d_model)
        self.dag_key = nn.Linear(d_model, d_model)
        # CRITICAL FIX: Use 1.0 scale instead of sqrt(d_model)^-1
        # sqrt(512)^-1 = 0.044 dampens logits → structure signal lost
        # Use 1.0 to preserve signal magnitude; BCE can handle large logits
        self.dag_scale = 1.0

    def forward(self, base_samples, int_samples, target_row, int_mask, mcm_mask=None):
        """
        Input:
            mcm_mask: (B, N) or None. If present, 1.0 means this token is masked.
        """
        # Gradient Checkpointing Fix:
        # use_reentrant=False requires at least one input to have requires_grad=True.
        # Since our inputs are data (no grad), we inject a dummy tensor.
        dummy_tensor = torch.tensor(0.0, device=base_samples.device, requires_grad=True)

        # Pass 1: Initial Guess
        if self.grad_checkpoint and self.training:
            # Inputs to pass 1 don't require grad, so use_reentrant=False is mandatory for checkpoint to work
            deltas_1, log_sigma_1, logits_1, aux_1 = checkpoint(self._forward_pass, base_samples, int_samples, target_row, int_mask, mcm_mask, None, dummy_tensor, use_reentrant=False)
        else:
            deltas_1, log_sigma_1, logits_1, aux_1 = self._forward_pass(base_samples, int_samples, target_row, int_mask, mcm_mask, None, dummy_tensor)
        
        if mcm_mask is not None:
             B, N = base_samples.shape
             dummy_adj = torch.zeros(B, N, N, device=base_samples.device)
             # Return 0.0 aux loss for MCM
             # Return None for log_sigma in MCM pass? Or dummy.
             return deltas_1, dummy_logits, dummy_adj, None, 0.0
        
        # --- Causal Masking Logic ---
        # 1. Compute Mask from Pass 1 Logits
        # logits_1: (B, N, N) -> mask: (B, N, N) bias
        mask = self.causal_mask_net(logits_1)
        
        # 2. Expand Mask for Interleaved Tokens (if needed)
        # Tokens: (B, S, D). Mask: (B, S, S).
        # if interleaved: S = 2N. Mask is (B, N, N).
        # We need to map Node i -> Node j dependencies to Token blocks.
        if not self.ablation_no_interleaved:
            # Expand (B, N, N) -> (B, 2N, 2N)
            # Each element (i, j) becomes a 2x2 block
            mask = mask.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
            
        # 3. Create Head Mask? 
        # nn.MultiheadAttention expects (B*nhead, S, S) or (B, S, S).
        # We have (B, S, S). To broadcast to (B, H, S, S), we need (B, 1, S, S).
        attn_mask = mask.unsqueeze(1) # Float bias added to scores
        
        # Pass 2: Refinement (Recurrent) with Causal Mask
        # Broadcast deltas (B, N) to (B, Context, N) if base_samples has context dim
        if len(base_samples.shape) == 3:
             refined_base = base_samples + deltas_1.unsqueeze(1)
        else:
             refined_base = base_samples + deltas_1
             
        if self.grad_checkpoint and self.training:
            # Capture logits_2 for Iterative Structure Refinement
            deltas_2, log_sigma_2, logits_2, aux_2 = checkpoint(self._forward_pass, refined_base, int_samples, target_row, int_mask, None, attn_mask, dummy_tensor, use_reentrant=False)
        else:
            deltas_2, log_sigma_2, logits_2, aux_2 = self._forward_pass(refined_base, int_samples, target_row, int_mask, None, attn_mask, dummy_tensor)
        
        # --- Recursive Mask Refinement (Pass 2 -> Pass 3) ---
        # "Iterative Structure Refinement": Use structure from Pass 2 to guide Pass 3
        mask_2 = self.causal_mask_net(logits_2)
        if not self.ablation_no_interleaved:
            mask_2 = mask_2.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        attn_mask_2 = mask_2.unsqueeze(1)

        # Pass 3: Final Polish (using Refined Mask)
        if len(base_samples.shape) == 3:
             refined_base_2 = base_samples + deltas_2.unsqueeze(1)
        else:
             refined_base_2 = base_samples + deltas_2
             
        if self.grad_checkpoint and self.training:
             deltas_final, log_sigma_final, logits_final, aux_3 = checkpoint(self._forward_pass, refined_base_2, int_samples, target_row, int_mask, None, attn_mask_2, dummy_tensor, use_reentrant=False)
        else:
             deltas_final, log_sigma_final, logits_final, aux_3 = self._forward_pass(refined_base_2, int_samples, target_row, int_mask, None, attn_mask_2, dummy_tensor)
        
        
        # Returning Dummy Logits/Adj for API compatibility with main.py metrics
        if len(base_samples.shape) == 3:
            B, S, N = base_samples.shape
        else:
            B, N = base_samples.shape
            
        # ISSUE 15: Return estimated adjacency instead of zeros
        # This provides useful debug info and satisfies "Predicted Adjacency" expectation.
        dummy_adj = torch.sigmoid(logits_final)
        
        # Total Aux Loss (Average or Sum?) - Sum encourages all steps to be balanced.
        total_aux = aux_1 + aux_2 + aux_3
        
        # Return log_sigma from final pass (logits_final comes from pass 3? No, logic is tricky)
        # _forward_pass returns (deltas, log_sigma, logits, aux)
        # In Pass 3: deltas_final, log_sigma_final, logits_final, aux_3
        # Variable name in code (lines 596): deltas_final, _, logits_final, aux_3
        # I need to capture log_sigma_final
        
        # Note: I need to update lines 539, 578, 596 to capture log_sigma!
        # But wait, multi_replace cannot update disparate lines easily if they are not targeted.
        # I will update the return here, but I must also update the CALL sites.
        
        return deltas_final, logits_final, dummy_adj, log_sigma_final, total_aux

    @torch.no_grad()
    def rollout_counterfactual(self, start_state, int_samples, int_mask, steps=5):
        """
        Generates a counterfactual trajectory (Generative World Model).
        
        Args:
            start_state: (B, N) Initial state (t=0)
            int_samples: (B, N) Intervention values (Target)
            int_mask: (B, N) Intervention mask
            steps: int (Number of rollout steps)
            
        Returns:
            trajectory: (B, Steps+1, N) - [X_0, X_1, ..., X_steps]
        """
        trajectory = [start_state]
        current_state = start_state.clone()
        
        # We assume target_row (Context) updates with current_state for autoregression
        # Or we can keep it fixed if it represents "Initial Conditions".
        # For a dynamical system x_t+1 = f(x_t), context is x_t.
        
        for _ in range(steps):
            # Pass current state as base AND target_row
            deltas, _, _, _, _ = self.forward(
                base_samples=current_state, 
                int_samples=int_samples, 
                target_row=current_state, 
                int_mask=int_mask
            )
            
            # Apply physics: x_{t+1} = x_t + delta
            current_state = current_state + deltas
            trajectory.append(current_state.clone())
            
        return torch.stack(trajectory, dim=1)

    def _forward_pass(self, base_samples, int_samples, target_row, int_mask, mcm_mask, attn_mask=None, dummy_arg=None):
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
        # Unpack pos_ids from encoder (Robust RoPE)
        x, pos_ids = self.encoder(base_samples, int_samples, enc_target, enc_int_mask)
        x, aux_loss = self.transformer(x, pos_ids=pos_ids, attn_mask=attn_mask)
        
        # Extract Value Tokens (embedding of the variable values)
        if self.ablation_no_interleaved:
            # Mode 'additive': All tokens are value tokens
            value_tokens = x
        else:
            # Mode 'interleaved': [ID, Value, ID, Value...]
            value_tokens = x[:, 1::2, :]   # (B, N, D)
            
        # Physics Head: Simple projection
        if self.ablation_no_physics:
             # Return zeros
             deltas = torch.zeros(value_tokens.shape[0], value_tokens.shape[1], device=value_tokens.device)
             log_sigma = torch.zeros_like(deltas)
        else:
             deltas = self.physics_head(value_tokens).squeeze(-1)  # (B, N, D) -> (B, N)
             # Predict log_sigma for I-NLL. (B, N)
             log_sigma = self.uncertainty_head(value_tokens).squeeze(-1)
        

            
        # DAG Prediction (Phase 5)
        logits = torch.zeros(value_tokens.shape[0], value_tokens.shape[1], value_tokens.shape[1], device=value_tokens.device)
        if not self.ablation_no_dag:
            # Query: "Who are my parents?"
            Q = self.dag_query(value_tokens) # (B, N, D)
            # Key: "I am a potential parent"
            K = self.dag_key(value_tokens)   # (B, N, D)
            
            logits = torch.matmul(Q, K.transpose(-2, -1)) * self.dag_scale
        
        # Return log_sigma in the 2nd slot (formerly mcm_out)
        return deltas, log_sigma, logits, aux_loss

    def anneal_temperature(self, epoch, total_epochs):
        # No tau needed for Gumbel (Hard=True)
        return 1.0
# --- Helper for Differentiable AllReduce ---
class DifferentiableAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Must be in-place for NCCL, but we want to return a new tensor to keep graph clean? 
        # Actually all_reduce is in-place.
        # We can implement it as: x is input.
        # But modifying x in place with autograd is tricky.
        # Better: Clone x, reduce clone, return clone.
        # To avoid "autograd kernel not registered" warning, 
        # operations here should not be tracked unless we use save_for_backward.
        
        if torch.distributed.is_initialized():
            x_reduced = x.clone() 
            # all_reduce is in-place on x_reduced.
            # DifferentiableAllReduce is a Function, so this forward runs in no_grad mode (usually).
            # But x_reduced inherits properties? verification needed.
            # "The forward method... run in a torch.no_grad() context automatically."
            # So x_reduced.requires_grad is False. Safe to all_reduce.
            torch.distributed.all_reduce(x_reduced, op=torch.distributed.ReduceOp.SUM)
            return x_reduced
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient of Sum(x_1...x_N) w.r.t x_i is just 1.0 * grad_output
        return grad_output

def differentiable_all_reduce_sum(x):
    return DifferentiableAllReduce.apply(x)
