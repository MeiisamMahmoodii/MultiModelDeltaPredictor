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
                   lambda_delta=10.0, lambda_dag=1.0, lambda_h=0.1, lambda_l1=0.01):
    loss_delta = nn.functional.huber_loss(pred_delta, true_delta)
    loss_dag = nn.functional.binary_cross_entropy_with_logits(pred_adj, true_adj)
    
    pred_prob = torch.sigmoid(pred_adj)
    batch_h = []
    # Loop over batch for trace calc
    for i in range(pred_adj.shape[0]):
        batch_h.append(compute_h_loss(pred_prob[i]))
    loss_h = torch.stack(batch_h).mean()
    
    loss_l1 = torch.norm(pred_prob, p=1) / pred_adj.numel()
    
    total_loss = (lambda_delta * loss_delta + 
                  lambda_dag * loss_dag + 
                  lambda_h * loss_h + 
                  lambda_l1 * loss_l1)
    
    return total_loss, {"delta": loss_delta.item(), "dag": loss_dag.item(), "h": loss_h.item()}
