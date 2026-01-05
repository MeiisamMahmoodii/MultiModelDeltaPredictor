import torch
import pytest
from src.training.metrics import compute_shd, compute_f1, compute_tpr_fdr

def test_shd_basic():
    # True: 0->1
    # Pred: 0->1, 1->0 (Extra edge)
    true_adj = torch.tensor([[[0., 1.],
                              [0., 0.]]])
    pred_logits = torch.tensor([[[0., 10.],  # High logit -> 1
                                 [10., -10.]]]) # High logit -> 1 (Wrong), Low -> 0
    
    # SHD should be 1 (Reverse? No, SHD counts 1 for extra, 1 for missing. 
    # Here True has 0->1. Pred has 0->1 and 1->0.
    # So Pred has 1 extra edge. SHD = 1.
    shd = compute_shd(pred_logits, true_adj)
    assert shd == 1.0

def test_shd_missing():
    # True: 0->1
    # Pred: Empty
    true_adj = torch.tensor([[[0., 1.],
                              [0., 0.]]])
    pred_logits = torch.tensor([[[ -10., -10.],
                                 [ -10., -10.]]])
    shd = compute_shd(pred_logits, true_adj)
    assert shd == 1.0

def test_f1_perfect():
    true_adj = torch.tensor([[[0., 1.], [0., 0.]]])
    pred_logits = torch.tensor([[[-10., 10.], [-10., -10.]]])
    f1 = compute_f1(pred_logits, true_adj)
    assert f1 == 1.0

def test_tpr_fdr():
    # True: 1 edge (0->1).
    # Pred: 1 edge (0->1) Correct.
    # TPR = TP/P = 1/1 = 1.
    # FDR = FP / (TP+FP) = 0 / 1 = 0.
    true_adj = torch.tensor([[[0., 1.], [0., 0.]]])
    pred_logits = torch.tensor([[[-10., 10.], [-10., -10.]]])
    
    tpr, fdr = compute_tpr_fdr(pred_logits, true_adj)
    assert tpr == 1.0
    assert fdr == 0.0

from src.training.metrics import compute_sid

def test_sid_metric():
    # Case 1: Identical graphs -> SID = 0
    # A->B
    true_adj = torch.tensor([[0., 1.], [0., 0.]])
    pred_logits = torch.tensor([[-10., 10.], [-10., -10.]])
    sid = compute_sid(pred_logits, true_adj)
    assert sid == 0.0
    
    # Case 2: Reverse edge
    # True: A->B
    # Pred: B->A
    # For pair (A, B):
    # Pred Parents(A) = {B}.
    # Is {B} a valid adjustment set for A->B in True Graph?
    # True Descendants(A) = {B}.
    # {B} is a descendant! So Cond 1 fails. SID should increase.
    # Count pairs: (A, B) -> Failed.
    # (B, A) -> Pred Parents(B) = {}.
    # True: A->B. Backdoor graph for B (remove B->...): A->B.
    # Is A d-separated from B given {}? No, A->B exists.
    # Wait, A->B is an incoming edge to B?
    # Backdoor graph for B: Remove outgoing B->... (None).
    # Graph is A->B.
    # Path B ... A?
    # B <- A.
    # d-sep(B, A, {}) in A->B?
    # B has collider? No.
    # It's simply A->B. Path exists.
    # So SID count >= 1.
    
    pred_logits_rev = torch.tensor([[-10., -10.], [10., -10.]])
    sid = compute_sid(pred_logits_rev, true_adj)
    assert sid > 0.0
