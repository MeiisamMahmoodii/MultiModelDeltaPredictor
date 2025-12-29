import torch
import torch.nn as nn
import torch.nn.functional as F
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

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = torch.mean(x.pow(2), dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm + self.eps)
        return self.scale * x_normed

class SwiGLUResBlock(nn.Module):
    """
    Residual Block with SwiGLU activation (LLaMA style).
    Input -> RMSNorm -> SwiGLU -> Linear -> Residual
    """
    def __init__(self, d_model, expansion_factor=4):
        super().__init__()
        dim_hidden = int(d_model * expansion_factor)
        
        self.norm = RMSNorm(d_model)
        
        # SwiGLU Mechanism
        self.w_gate = nn.Linear(d_model, dim_hidden, bias=False)
        self.w_val = nn.Linear(d_model, dim_hidden, bias=False)
        self.w_out = nn.Linear(dim_hidden, d_model, bias=False)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        # SwiGLU: (Gate * Val)
        gate = F.silu(self.w_gate(x))
        val = self.w_val(x)
        out = self.w_out(gate * val)
        
        return residual + out

class DeepCausalExpert(nn.Module):
    """
    A deep expert network composed of multiple SwiGLU blocks.
    Represents a specific "physical law" or causal mechanism.
    """
    def __init__(self, d_model, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            SwiGLUResBlock(d_model) for _ in range(num_layers)
        ])
        self.final_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (Batch, d_model)
        for layer in self.layers:
            x = layer(x)
        return self.final_proj(x) # (Batch, 1)

class MoELayer(nn.Module):
    """
    Mixture of Experts Layer.
    Pool of experts shared across nodes.
    Each node has a router to pick weights for experts.
    """
    def __init__(self, d_model, num_experts=8, num_layers_per_expert=4, num_nodes=55):
        super().__init__()
        self.num_experts = num_experts
        
        # 1. Shared Pool of Experts
        self.experts = nn.ModuleList([
            DeepCausalExpert(d_model, num_layers=num_layers_per_expert)
            for _ in range(num_experts)
        ])
        
        # 2. Per-Node Routers (Classifiers)
        # Input: d_model (Context) -> Output: num_experts (Logits)
        self.routers = nn.ModuleList([
            nn.Linear(d_model, num_experts) for _ in range(num_nodes)
        ])
        
    def forward(self, x, node_indices=None):
        """
        x: (Batch, Num_Nodes_Active, d_model) - Context for each node
        node_indices: List of actual node indices [0, 1, 2...] corresponding to columns of x
        """
        batch_size, num_active, d_model = x.shape
        final_deltas = []
        
        # For each active node in the sequence
        for i in range(num_active):
            node_idx = i # In standard training, index matches position. 
            # If standard, node_indices[i] would be safer if we shuffled.
            # Assuming standard ordering 0..N for now:
            
            # 1. Get Router Logits
            # router[node_idx] gives us the preference for this physical variable
            router_logits = self.routers[node_idx](x[:, i, :]) # (Batch, num_experts)
            
            # 2. Top-K Gating (K=2)
            # Softmax
            weights = F.softmax(router_logits, dim=-1)
            
            # Top 2
            top_k_weights, top_k_indices = torch.topk(weights, k=2, dim=-1) # (Batch, 2)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True) # Re-normalize
            
            # 3. Apply Experts
            # This is the slow "loop" way, optimized implementations use scatter/gather.
            # Given small scale (Batch=32, Experts=8), loop is fine for readability.
            
            # We need to compute: sum(weight_j * Expert_j(input))
            # But doing it efficiently is tricky.
            # Let's run ALL experts on the input? (Wasteful but simple for v1)
            # Actually, let's just run the Top-2.
            
            node_output = torch.zeros(batch_size, 1, device=x.device)
            
            # Iterate over the K selected experts
            for k in range(2):
                idx = top_k_indices[:, k] # (Batch,) indices of expert for each sample
                w = top_k_weights[:, k].unsqueeze(1) # (Batch, 1)
                
                # We have mixed experts per batch. 
                # Ideally, we mask.
                
                # Simplified Approach for V1:
                # "Soft MoE": Just run all 8 experts, weighted average.
                # It's computationally heavy but exact and easy to code.
                # Then we can sparsify later.
                pass
            
            # Re-implementation: Soft MoE (Weight all 8)
            # For correctness and simplicity in "Model G" phase.
            node_experts_out = torch.zeros(batch_size, 1, device=x.device)
            
            # Check if we should optimization
            # Vectorized run of all experts?
            # Creating a stack of outputs: (Batch, Num_Experts, 1)
            
            # Only run the chosen ones? 
            # Vectorization is hard with dynamic experts per sample.
            # Let's do the "Dense Gating" (Soft MoE) for now.
            # It acts like an ensemble.
            
            expert_outputs = []
            for exp in self.experts:
                expert_outputs.append(exp(x[:, i, :])) # (Batch, 1)
            expert_outputs = torch.cat(expert_outputs, dim=1) # (Batch, Num_Experts)
            
            # Weighted Sum
            delta_pred = (expert_outputs * weights).sum(dim=1, keepdim=True) # (Batch, 1)
            final_deltas.append(delta_pred)
            
        return torch.cat(final_deltas, dim=-1) # (Batch, Num_Active)

class CausalTransformer(nn.Module):
    """
    Unified Causal Transformer (Model G - Universal Causal MoE)
    Features:
    - Interleaved Tokenization
    - Shared Mixture of Experts (SwiGLU + Residual)
    - Sparse Structure Learning
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
        
        # 4. Universal MoE Layer
        self.moe = MoELayer(d_model, num_experts=8, num_layers_per_expert=4, num_nodes=num_nodes)
        
        # 5. Sparse Structure Learning
        self.dag_parent = nn.Linear(d_model, d_model)
        self.dag_child = nn.Linear(d_model, d_model)
        self.tau = 1.0

    def forward(self, base_samples, int_samples, target_row, int_mask, int_node_idx):
        # Encodings: (B, 2N, D)
        x = self.encoder(base_samples, int_samples, target_row, int_mask)
        x = self.transformer(x) 
        
        feature_tokens = x[:, 0::2, :] # (B, N, D)
        value_tokens = x[:, 1::2, :]   # (B, N, D)
        
        # --- Hyper-Expert Delta Prediction ---
        instr = self.int_embedding(int_node_idx) # (B, 32)
        modulation = self.hyper_net(instr).unsqueeze(1) # (B, 1, d_model)
        
        # Modulate Value Tokens ("The state has changed rules")
        x_modulated = value_tokens * modulation
        
        # Pass through MoE Layer
        deltas = self.moe(x_modulated)
        
        # --- Structural Discovery ---
        p = self.dag_parent(feature_tokens)
        c = self.dag_child(feature_tokens)
        logits = torch.matmul(p, c.transpose(-2, -1))
        
        adj_sampled = gumbel_sigmoid(logits, tau=self.tau, hard=self.training)
        
        return deltas, logits, adj_sampled

    def anneal_temperature(self, epoch, total_epochs):
        self.tau = max(0.1, 1.0 - (epoch / total_epochs))
        return self.tau
