import torch
from sklearn.metrics import f1_score

def compute_shd(pred_adj_logits, true_adj_matrix, threshold=0.0):
    """
    Computes Structural Hamming Distance (SHD).
    """
    with torch.no_grad():
        pred_edges = (pred_adj_logits > threshold).float()
        true_edges = true_adj_matrix.float()
        diff = torch.abs(pred_edges - true_edges)
        shd = diff.sum(dim=(1, 2))
        return shd.mean().item()

def compute_f1(pred_adj_logits, true_adj_matrix, threshold=0.0):
    """
    Computes F1 Score for edges.
    """
    with torch.no_grad():
        pred_prob = torch.sigmoid(pred_adj_logits)
        pred_flat = (pred_prob > 0.5).cpu().numpy().flatten()
        true_flat = true_adj_matrix.cpu().numpy().flatten()
        return f1_score(true_flat, pred_flat, zero_division=0)

def compute_mae(pred_delta, true_delta):
    """
    Computes Mean Absolute Error for deltas.
    """
    with torch.no_grad():
        return torch.nn.functional.l1_loss(pred_delta, true_delta).item()

def compute_tpr_fdr(pred_adj_logits, true_adj_matrix, threshold=0.0):
    """
    Computes True Positive Rate (Recall) and False Discovery Rate.
    """
    with torch.no_grad():
        pred_edges = (pred_adj_logits > threshold).float()
        true_edges = true_adj_matrix.float()
        
        tp = (pred_edges * true_edges).sum()
        fp = (pred_edges * (1 - true_edges)).sum()
        fn = ((1 - pred_edges) * true_edges).sum()
        
        tpr = tp / (tp + fn + 1e-8)
        fdr = fp / (tp + fp + 1e-8)
        
        return tpr.item(), fdr.item()
