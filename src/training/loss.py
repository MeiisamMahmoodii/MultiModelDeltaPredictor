import torch
import torch.nn as nn

def compute_h_loss(adj_matrix):
    N = adj_matrix.shape[-1]
    # Matrix Exp is not implemented for MPS, so we fallback to CPU for this op
    if adj_matrix.device.type == 'mps':
        A_sq = (adj_matrix * adj_matrix).cpu()
        h = torch.trace(torch.matrix_exp(A_sq)) - N
        return h.to(adj_matrix.device)
    else:
        A_sq = adj_matrix * adj_matrix
        h = torch.trace(torch.matrix_exp(A_sq)) - N
        return h

def causal_loss_fn(pred_delta, true_delta, pred_adj, true_adj, 
                   lambda_delta=100.0, lambda_dag=0.0, lambda_h=0.0, lambda_l1=0.0):
    loss_delta = nn.functional.huber_loss(pred_delta, true_delta)
    
    # Phase 4: Decoupled Learning (No DAG Head)
    # We ignore pred_adj and true_adj for now
    loss_dag = torch.tensor(0.0, device=pred_delta.device)
    loss_h = torch.tensor(0.0, device=pred_delta.device)
    
    total_loss = loss_delta * lambda_delta
    
    return total_loss, {"delta": loss_delta.item(), "dag": 0.0, "h": 0.0}

def mcm_loss_fn(pred_values, true_values, mask_indices):
    """
    Masked Causal Modeling Loss.
    """
    # Only calculate loss on masked tokens
    loss = nn.functional.mse_loss(pred_values[mask_indices], true_values[mask_indices])
    return loss
