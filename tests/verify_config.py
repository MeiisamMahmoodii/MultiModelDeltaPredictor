
import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.CausalTransformer import CausalTransformer
from src.data.SCMGenerator import SCMGenerator
from src.data.CausalDataset import CausalDataset
from src.data.collate import collate_fn_pad
from torch.utils.data import DataLoader

def test_model_depth():
    print("--- Testing Model Depth Configuration ---")
    num_layers = 24
    d_model = 512
    num_nodes = 55 # 50 vars + 5 buffer
    
    print(f"Initializing CausalTransformer with num_layers={num_layers}, d_model={d_model}...")
    model = CausalTransformer(num_nodes=num_nodes, d_model=d_model, num_layers=num_layers)
    
    actual_layers = len(model.transformer.layers)
    print(f"Actual Transformer Layers: {actual_layers}")
    
    if actual_layers == num_layers:
        print("✅ Depth Check Passed")
    else:
        print(f"❌ Depth Check Failed: Expected {num_layers}, got {actual_layers}")
        exit(1)

def test_fixed_vars():
    print("\n--- Testing Fixed Variable Count (50) ---")
    min_vars = 50
    max_vars = 50
    
    print(f"Initializing Generator with min_vars={min_vars}, max_vars={max_vars}...")
    gen = SCMGenerator(num_nodes=max_vars, edge_prob=0.2)
    
    dataset = CausalDataset(
        gen, 
        num_nodes_range=(min_vars, max_vars),
        samples_per_graph=4,
        infinite=False,
        validation_graphs=1
    )
    
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn_pad)
    
    batch = next(iter(dataloader))
    target = batch['target_row']
    
    # Check shape: (Batch, Num_Vars)
    print(f"Batch Target Shape: {target.shape}")
    
    if target.shape[1] == 50:
        print("✅ Variable Count Check Passed")
    else:
        print(f"❌ Variable Count Check Failed: Expected 50, got {target.shape[1]}")
        exit(1)
        
    return batch

def test_forward_pass(batch):
    print("\n--- Testing Forward Pass with Deep Model & 50 Vars ---")
    num_layers = 24
    d_model = 512
    max_vars = 50
    
    model = CausalTransformer(num_nodes=max_vars + 5, d_model=d_model, num_layers=num_layers)
    
    # Move to CPU for test
    device = torch.device('cpu')
    model.to(device)
    
    base = batch['base_samples']
    int_s = batch['int_samples']
    target = batch['target_row']
    mask = batch['int_mask']
    idx = batch['int_node_idx'] # Should be None/Ignored in Phase 3 but still passed? 
    # Actually the collate might pass it, let's check
    
    print("Running forward pass...")
    try:
        deltas, logits, adj, _, _ = model(base, int_s, target, mask)
        print(f"Output Shapes - Delta: {deltas.shape}, Logits: {logits.shape}, Adj: {adj.shape}")
        print("✅ Forward Pass Passed")
    except Exception as e:
        print(f"❌ Forward Pass Failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    test_model_depth()
    batch = test_fixed_vars()
    test_forward_pass(batch)
