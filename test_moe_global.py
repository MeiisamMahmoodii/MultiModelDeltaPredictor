import torch
import torch.distributed as dist
from unittest.mock import MagicMock, patch
import src.models.CausalTransformer as CT
from src.models.CausalTransformer import SimpleMoELayer

print(f"DEBUG: Loading CausalTransformer from {CT.__file__}")
print(f"DEBUG: SimpleMoELayer defined in {SimpleMoELayer.__module__}")
import inspect
print(f"DEBUG: SimpleMoELayer source: {inspect.getsourcefile(SimpleMoELayer)}")

def test_single_gpu():
    print("\n--- Test Single GPU (Local) ---")
    d_model = 128
    layer = SimpleMoELayer(d_model=d_model, num_experts=4)
    x = torch.randn(2, 10, d_model) # Batch=2, Seq=10
    
    # Run forward
    out, aux_loss = layer(x)
    print(f"Output shape: {out.shape}")
    print(f"Aux Loss: {aux_loss.item()}")
    
    assert out.shape == (2, 10, d_model)
    assert not torch.isnan(aux_loss)

def test_distributed_mock():
    print("\n--- Test Distributed (Mocked) ---")
    
    # Mock distributed
    with patch('torch.distributed.is_initialized', return_value=True):
        with patch('torch.distributed.all_reduce') as mock_all_reduce:
            
            # Define side effect for all_reduce: multiply by 2 (Simulate 2 identical GPUs)
            def side_effect(tensor, op=None):
                tensor.mul_(2.0)
                return None
            mock_all_reduce.side_effect = side_effect
            
            d_model = 128
            layer = SimpleMoELayer(d_model=d_model, num_experts=4)
            x = torch.randn(2, 10, d_model)
            
            # Run forward
            out, aux_loss = layer(x)
            
            # Verify all_reduce called twice (sum of probs + count)
            print(f"DEBUG: all_reduce.call_count = {mock_all_reduce.call_count}")
            assert mock_all_reduce.call_count == 2
            print("all_reduce called successfully.")
            print(f"Aux Loss (Global): {aux_loss.item()}")

if __name__ == "__main__":
    test_single_gpu()
    test_distributed_mock()
    print("\nAll Tests Passed!")
