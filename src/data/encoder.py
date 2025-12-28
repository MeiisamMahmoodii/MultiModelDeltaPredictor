import torch
import torch.nn as nn

class InterleavedEncoder(nn.Module):
    """
    Transforms Causal Data (Base, Mask, Value) into Interleaved Tokens.
    Format: [Feature_ID_0, Value_0, Feature_ID_1, Value_1, ...]
    """
    def __init__(self, num_vars, d_model):
        super().__init__()
        self.num_vars = num_vars
        self.d_model = d_model
        
        # Embeddings
        # 1. Feature ID Embedding: "Who am I?"
        self.var_id_emb = nn.Embedding(num_vars + 1, d_model)
        
        # 2. Value Embedding: "What is my value?"
        # We use a simple projection for continuous values
        # In TabPFN, they use a more complex MLP, but linear is fine for v1
        self.value_emb = nn.Linear(1, d_model)
        
        # 3. Type Embedding: "Am I Observed or Intervened?"
        self.type_emb = nn.Embedding(2, d_model) # 0=Obs, 1=Int

    def forward(self, base_samples, int_samples, target_row, int_mask):
        """
        Args:
            target_row: (B, N) - The sample we are processing
            int_mask: (B, N) - 1.0 where intervened
        Returns:
            tokens: (B, 2N, D)
        """
        B, N = target_row.shape
        device = target_row.device
        
        # 1. Feature Tokens
        # Create IDs: [0, 1, ..., N-1] repeated B times
        ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1) # (B, N)
        f_emb = self.var_id_emb(ids) # (B, N, D)
        
        # 2. Value Tokens
        # Embed the scalar values
        values = target_row.unsqueeze(-1) # (B, N, 1)
        v_emb = self.value_emb(values) # (B, N, D)
        
        # Add Type Information to Value Tokens
        # If int_mask=1, we add 'Intervention' embedding
        t_ids = int_mask.long() # (B, N)
        t_emb = self.type_emb(t_ids) # (B, N, D)
        v_emb = v_emb + t_emb
        
        # 3. Interleave
        # Stack: (B, N, 2, D) -> Flatten: (B, 2N, D)
        stacked = torch.stack([f_emb, v_emb], dim=2)
        tokens = stacked.flatten(1, 2)
        
        return tokens
