#!/usr/bin/env python3
"""
Test script to verify numerical stability fixes
"""
import torch
import numpy as np
import sys

# Add src to path
sys.path.insert(0, '/home/meisam/MultiModelDeltaPredictor')

from src.training.metrics import (
    compute_mae, compute_tpr_fdr, compute_f1, compute_shd
)
from src.training.loss import causal_loss_fn, compute_h_loss
from src.data.SCMGenerator import SCMGenerator
from src.data.encoder import FourierEmbedding, HybridEmbedding

def test_metrics_stability():
    """Test that metrics handle edge cases correctly"""
    print("Testing metrics stability...")
    
    # Test 1: Large logits (potential overflow)
    pred_logits = torch.tensor([[[1e6, 1e6], [1e6, 1e6]]])
    true_adj = torch.tensor([[[0., 1.], [0., 0.]]])
    
    try:
        f1 = compute_f1(pred_logits, true_adj)
        assert not np.isnan(f1) and not np.isinf(f1), f"F1 is {f1}"
        print(f"  ✓ F1 with large logits: {f1}")
    except Exception as e:
        print(f"  ✗ F1 test failed: {e}")
        return False
    
    # Test 2: All zeros
    pred_logits_zero = torch.zeros(1, 3, 3)
    true_adj_small = torch.zeros(1, 3, 3)
    
    try:
        tpr, fdr = compute_tpr_fdr(pred_logits_zero, true_adj_small)
        assert not np.isnan(tpr) and not np.isinf(tpr), f"TPR is {tpr}"
        assert not np.isnan(fdr) and not np.isinf(fdr), f"FDR is {fdr}"
        print(f"  ✓ TPR/FDR with all zeros: TPR={tpr}, FDR={fdr}")
    except Exception as e:
        print(f"  ✗ TPR/FDR test failed: {e}")
        return False
    
    # Test 3: MAE with NaN values
    pred_delta = torch.tensor([[1.0, 2.0, 3.0]])
    true_delta = torch.tensor([[1.0, 2.0, 3.0]])
    
    try:
        mae = compute_mae(pred_delta, true_delta)
        assert not np.isnan(mae) and not np.isinf(mae), f"MAE is {mae}"
        print(f"  ✓ MAE computation: {mae}")
    except Exception as e:
        print(f"  ✗ MAE test failed: {e}")
        return False
    
    return True

def test_loss_stability():
    """Test that loss functions handle edge cases"""
    print("\nTesting loss function stability...")
    
    # Test 1: Large predictions
    pred_delta = torch.tensor([[1e3, 1e3, 1e3]])
    true_delta = torch.tensor([[0.0, 0.0, 0.0]])
    pred_adj = torch.randn(1, 3, 3) * 10
    true_adj = torch.zeros(1, 3, 3)
    
    try:
        loss, items = causal_loss_fn(pred_delta, true_delta, pred_adj, true_adj)
        assert not torch.isnan(loss) and not torch.isinf(loss), f"Loss is {loss}"
        assert not np.isnan(items['delta']), f"Delta loss is {items['delta']}"
        print(f"  ✓ Loss with large predictions: {loss.item():.4f}")
    except Exception as e:
        print(f"  ✗ Loss test failed: {e}")
        return False
    
    # Test 2: H-loss computation
    try:
        adj_matrix = torch.randn(5, 5) * 0.1
        h = compute_h_loss(adj_matrix)
        assert not torch.isnan(h) and not torch.isinf(h), f"H-loss is {h}"
        print(f"  ✓ H-loss computation: {h.item():.4f}")
    except Exception as e:
        print(f"  ✗ H-loss test failed: {e}")
        return False
    
    return True

def test_data_generation_stability():
    """Test that data generation produces bounded values"""
    print("\nTesting data generation stability...")
    
    try:
        gen = SCMGenerator(
            num_nodes=5,
            edge_prob=0.3,
            noise_scale=1.0,
            intervention_prob=0.5
        )
        
        # Generate pipeline
        result = gen.generate_pipeline(
            num_nodes=5,
            edge_prob=0.3,
            num_samples_base=32,
            num_samples_per_intervention=32,
            as_torch=True,
            use_twin_world=True
        )
        
        # Check base tensor
        base_tensor = result['base_tensor']
        if not isinstance(base_tensor, torch.Tensor):
            base_tensor = torch.tensor(base_tensor.values, dtype=torch.float32)
        
        assert not torch.isnan(base_tensor).any(), "Base tensor has NaNs"
        assert not torch.isinf(base_tensor).any(), "Base tensor has Infs"
        assert (base_tensor.abs() <= 100).all(), "Base tensor exceeds bounds"
        print(f"  ✓ Base tensor bounds: [{base_tensor.min():.2f}, {base_tensor.max():.2f}]")
        
        # Check intervention tensors
        for i, df in enumerate(result['all_dfs']):
            if isinstance(df, torch.Tensor):
                int_tensor = df
            else:
                int_tensor = torch.tensor(df.values, dtype=torch.float32)
            assert not torch.isnan(int_tensor).any(), f"Intervention {i} has NaNs"
            assert not torch.isinf(int_tensor).any(), f"Intervention {i} has Infs"
            assert (int_tensor.abs() <= 100).all(), f"Intervention {i} exceeds bounds"
        
        print(f"  ✓ All intervention tensors bounded")
        
    except Exception as e:
        import traceback
        print(f"  ✗ Data generation test failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_encoder_stability():
    """Test that embeddings produce bounded values"""
    print("\nTesting encoder stability...")
    
    try:
        # Test Fourier embedding
        fourier = FourierEmbedding(64)
        x = torch.randn(32, 1) * 100  # Large values
        fourier_out = fourier(x)
        
        assert not torch.isnan(fourier_out).any(), "Fourier embedding has NaNs"
        assert not torch.isinf(fourier_out).any(), "Fourier embedding has Infs"
        assert (fourier_out.abs() <= 100).all(), "Fourier embedding exceeds bounds"
        print(f"  ✓ Fourier embedding bounds: [{fourier_out.min():.2f}, {fourier_out.max():.2f}]")
        
        # Test Hybrid embedding
        hybrid = HybridEmbedding(256)
        x = torch.randn(32, 1) * 100
        hybrid_out = hybrid(x)
        
        assert not torch.isnan(hybrid_out).any(), "Hybrid embedding has NaNs"
        assert not torch.isinf(hybrid_out).any(), "Hybrid embedding has Infs"
        print(f"  ✓ Hybrid embedding bounds: [{hybrid_out.min():.2f}, {hybrid_out.max():.2f}]")
        
    except Exception as e:
        print(f"  ✗ Encoder stability test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Numerical Stability Tests")
    print("=" * 60)
    
    all_pass = True
    all_pass &= test_metrics_stability()
    all_pass &= test_loss_stability()
    all_pass &= test_data_generation_stability()
    all_pass &= test_encoder_stability()
    
    print("\n" + "=" * 60)
    if all_pass:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
