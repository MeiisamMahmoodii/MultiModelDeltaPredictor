import torch
import sys
sys.path.append('.')

from src.models.CausalTransformer import CausalTransformer

def test_generative():
    print("\n--- Test Generative Capabilities ---")
    
    # 1. Setup Model
    model = CausalTransformer(num_nodes=10, d_model=64, num_layers=2)
    model.eval()
    
    B = 2
    N = 10
    base_samples = torch.randn(B, N)
    int_samples = torch.randn(B, N)
    target_row = torch.randn(B, N)
    int_mask = torch.zeros(B, N)
    int_mask[:, 0] = 1.0
    
    # 2. Test Forward Return Signature
    print("Testing Forward Pass...")
    ret = model(base_samples, int_samples, target_row, int_mask)
    # Expected: deltas, logits, dummy_adj, log_sigma, aux
    assert len(ret) == 5, f"Expected 5 returns, got {len(ret)}"
    
    deltas = ret[0]
    log_sigma = ret[3]
    
    print(f"Deltas shape: {deltas.shape}")
    print(f"Log Sigma shape: {log_sigma.shape}")
    
    assert deltas.shape == (B, N)
    assert log_sigma.shape == (B, N)
    print("Forward pass confirmed (log_sigma returned).")
    
    # 3. Test Rollout
    print("Testing Rollout...")
    steps = 5
    traj = model.rollout_counterfactual(base_samples, int_samples, int_mask, steps=steps)
    print(f"Trajectory shape: {traj.shape}")
    
    assert traj.shape == (B, steps+1, N)
    print("Rollout successful.")
    
    print("\nAll Generative Tests Passed.")

if __name__ == "__main__":
    test_generative()
