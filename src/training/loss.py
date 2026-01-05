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
    
    # Phase 5: Unified Learning (Structure Enabled)
    # 1. DAG Construction Loss (Binary Cross Entropy on Edges)
    # pred_adj are raw logits.
    loss_dag = nn.functional.binary_cross_entropy_with_logits(
        pred_adj, 
        true_adj,
        pos_weight=torch.tensor(3.0, device=pred_adj.device) # Fixed imbalance correction (3.0 for ~25% density)
    )
    
    # 2. Acyclicity Loss (H-Score)
    # We need Probabilities for H-score
    adj_prob = torch.sigmoid(pred_adj)
    # Reduce to [N, N] for H-score? Unbatching?
    # Actually compute_h_loss usually expects a single matrix or batched trace.
    # Our compute_h_loss implementation: A_sq = adj * adj.
    # If adj is (B, N, N), matmul works per batch.
    # Trace logic needs to handle batch dimension.
    
    # Let's check compute_h_loss implementation above. 
    # It does `torch.trace`. torch.trace only works on 2D tensors!
    # We need to average h over the batch.
    
    loss_h = 0.0
    if lambda_h > 0:
        # Loop over batch for safety or vectorize trace?
        # Einsum 'bii -> b' is trace.
        # But matrix_exp is expensive.
        # Use mean matrix? No, H(Mean(A)) != Mean(H(A)).
        # For efficiency, let's take mean of probabilities and enforce H on that?
        # "Consensus DAG": The batch likely comes from the SAME graph structure (if seeded identically)
        # BUT in our generator, every sample might be a different graph?
        # Wait, SCMGenerator main loop makes ONE graph per epoch step? No.
        # CausalDataset generates infinite random graphs.
        # So every sample in batch is a DIFFERENT graph.
        # We must penalize H for EACH graph.
        # This is very expensive (Matrix Exp for BxN).
        # Optimization: Only calculate H on a subset or accumulate?
        # Let's try iterating for now (Batch=16, 32 is small).
        
        h_sum = 0
        for i in range(len(adj_prob)):
             h_sum += compute_h_loss(adj_prob[i])
        loss_h = h_sum / len(adj_prob)

    loss_l1 = 0.0
    if lambda_l1 > 0:
        loss_l1 = torch.mean(torch.abs(adj_prob))
        
    total_loss = (loss_delta * lambda_delta) + (loss_dag * lambda_dag) + (loss_h * lambda_h) + (loss_l1 * lambda_l1)
    
    return total_loss, {
        "delta": loss_delta.item(), 
        "dag": loss_dag.item(), 
        "h": loss_h.item() if isinstance(loss_h, torch.Tensor) else loss_h
    }

def mcm_loss_fn(pred_values, true_values, mask_indices):
    """
    Masked Causal Modeling Loss.
    """
    # Only calculate loss on masked tokens
    loss = nn.functional.mse_loss(pred_values[mask_indices], true_values[mask_indices])
    return loss
