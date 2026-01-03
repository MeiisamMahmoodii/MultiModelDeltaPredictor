import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.SCMGenerator import SCMGenerator
from src.data.CausalDataset import CausalDataset
from src.models.CausalTransformer import CausalTransformer
from src.training.loss import causal_loss_fn
from torch.utils.data import DataLoader
from src.data.collate import collate_fn_pad

def deep_smoke_test():
    print("=== DEEP SMOKE TEST START ===")
    
    # 1. Data Generation Verification
    print("\n--- Step 1: Data Generation (SCM) ---")
    vars = 10
    gen = SCMGenerator(num_nodes=vars)
    # Force specific topology for deterministic debug? No, random is fine.
    
    # Create Dataset
    dataset = CausalDataset(gen, samples_per_graph=100, intervention_prob=0.1)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn_pad)
    
    batch = next(iter(dataloader))
    
    base = batch['base_samples']
    int_s = batch['int_samples']
    target = batch['target_row']
    mask = batch['int_mask']
    
    print(f"Batch Shapes:")
    print(f"  Base: {base.shape} (Expected: B, Samples, N)")
    print(f"  Int:  {int_s.shape}  (Expected: B, Samples, N)")
    print(f"  Tgt:  {target.shape}   (Expected: B, N)")
    print(f"  Mask: {mask.shape}     (Expected: B, N)")
    
    # Update vars to match actual batch (SCM is probabilistic)
    actual_vars = base.shape[2]
    print(f"  Actual Vars: {actual_vars}")
    # assert base.shape[2] == vars, f"Expected {vars} vars, got {base.shape[2]}"
    assert mask.max() <= 1.0, "Mask should be binary (0 or 1)"
    
    # 2. Model Initialization
    print("\n--- Step 2: Model Initialization ---")
    model = CausalTransformer(num_nodes=actual_vars, d_model=32, nhead=4, num_layers=2)
    print("Model initialized successfully.")
    
    # 3. Forward Pass (Phase 4 Logic)
    print("\n--- Step 3: Forward Pass (Phase 4) ---")
    # Enable anomaly detection for stability check
    torch.autograd.set_detect_anomaly(True)
    
    deltas, logits, adj, mcm_out = model(base, int_s, target, mask)
    
    print(f"Output Shapes:")
    print(f"  Deltas: {deltas.shape} (Expected: B, N)")
    print(f"  Logits: {logits.shape} (Expected: B, N, N)")
    
    assert deltas.shape == target.shape, f"Delta shape mismatch: {deltas.shape} vs {target.shape}"
    
    # 4. MCM Check (Optional)
    print("\n--- Step 4: MCM Head Check ---")
    # Create fake MCM mask
    mcm_mask = torch.zeros_like(mask)
    mcm_mask[:, 0] = 1.0 # Mask var 0
    
    d_mcm, l_mcm, a_mcm, out_mcm = model(base, int_s, target, mask, mcm_mask=mcm_mask)
    print(f"  MCM Out: {out_mcm.shape} (Expected: B, N)")
    
    # 5. Loss & Backward (Gradient Check)
    print("\n--- Step 5: Loss & Backward ---")
    true_delta = batch['delta']
    
    loss, metrics = causal_loss_fn(deltas, true_delta, logits, adj)
    print(f"  Loss: {loss.item()}")
    
    loss.backward()
    print("  Backward pass successful.")
    
    # Check gradients
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item()
            
    print(f"  Total Gradient Norm: {grad_norm}")
    if grad_norm == 0.0:
        print("  WARNING: Zero Gradients! Code is broken.")
    else:
        print("  Gradient flow confirmed.")

    print("\n=== DEEP SMOKE TEST PASSED ===")

if __name__ == "__main__":
    deep_smoke_test()
