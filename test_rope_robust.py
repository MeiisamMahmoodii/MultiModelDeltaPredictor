import torch
import sys
# Ensure src is in path
sys.path.append('.')

from src.models.CausalTransformer import CausalTransformer
from src.data.encoder import InterleavedEncoder

def test_rope_robust():
    print("\n--- Test Robust RoPE Flow ---")
    
    # 1. Setup Model
    num_nodes = 10
    d_model = 64
    model = CausalTransformer(num_nodes=num_nodes, d_model=d_model, num_layers=2, nhead=4)
    model.eval()
    
    # 2. Dummy Inputs
    B = 2
    base_samples = torch.randn(B, num_nodes)
    int_samples = torch.randn(B, num_nodes)
    target_row = torch.randn(B, num_nodes)
    int_mask = torch.zeros(B, num_nodes)
    int_mask[:, 0] = 1.0 # Intervene on node 0
    
    # 3. Test Encoder Output
    print("Testing Encoder...")
    tokens, pos_ids = model.encoder(base_samples, int_samples, target_row, int_mask)
    print(f"Tokens shape: {tokens.shape}") # Should be (B, 2*N, D)
    print(f"Pos IDs shape: {pos_ids.shape}") # Should be (B, 2*N)
    
    # Verify Pos IDs Logic
    # For interleaved, we expect [0, 0, 1, 1, 2, 2...]
    print(f"Sample Pos IDs (first 6): {pos_ids[0, :6].tolist()}")
    expected = [0, 0, 1, 1, 2, 2]
    assert pos_ids[0, :6].tolist() == expected, f"Expected {expected}, got {pos_ids[0, :6].tolist()}"
    
    # 4. Test Forward Pass (Propagation)
    print("Testing Full Forward...")
    out = model(base_samples, int_samples, target_row, int_mask)
    deltas = out[0]
    print(f"Deltas shape: {deltas.shape}")
    
    print("Test Passed: Forward pass successful with explicit pos_ids.")

if __name__ == "__main__":
    test_rope_robust()
