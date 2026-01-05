import torch
import pytest
from src.training.loss import compute_h_loss, causal_loss_fn

def test_h_loss_dag():
    # A->B->C (Acyclic)
    # 0->1, 1->2
    adj = torch.tensor([[0., 1., 0.],
                        [0., 0., 1.],
                        [0., 0., 0.]])
    h = compute_h_loss(adj)
    # Trace(exp(A*A)) - d should be 0 for acyclic
    # A*A: 
    # [0 1 0] * [0 1 0] = [0 0 1]
    # [0 0 1]   [0 0 1]   [0 0 0]
    # [0 0 0]   [0 0 0]   [0 0 0]
    # A*A*A = 0.
    # exp(M) = I + M + M^2/2...
    # Trace(I) = 3. Trace(M)=0. Trace(M^2)=0.
    # So exp(A*A) trace is 3. h = 3 - 3 = 0.
    assert torch.allclose(h, torch.tensor(0.0), atol=1e-5)

def test_h_loss_cycle():
    # A->B->A (Cycle)
    # 0->1, 1->0
    adj = torch.tensor([[0., 1.],
                        [1., 0.]])
    h = compute_h_loss(adj)
    # A^2 = I.
    # exp(A^2) = exp(I) = e*I = diag(e, e).
    # Trace = 2e.
    # h = 2e - 2 approx 2(2.718) - 2 = 3.43... > 0
    assert h.item() > 0.0

def test_causal_loss_function_structure():
    B, N = 2, 5
    pred_delta = torch.randn(B, N)
    true_delta = torch.randn(B, N)
    pred_adj = torch.randn(B, N, N) # Logits
    true_adj = torch.zeros(B, N, N)
    
    loss, details = causal_loss_fn(pred_delta, true_delta, pred_adj, true_adj, lambda_dag=1.0, lambda_h=1.0)
    
    assert isinstance(loss, torch.Tensor)
    assert "delta" in details
    assert "dag" in details
    assert "h" in details
