import torch
import torch.nn.functional as F

def collate_fn_pad(batch):
    """
    Pads and flattens a batch of graph chunks.
    Input: List of Dicts (Each dict contains tensors for ONE graph, shape (S, N))
    Output: Collated Dict (Concatenated tensors, shape (Total_S, Max_N))
    """
    # 1. Find Max Nodes in this batch
    max_nodes = 0
    for item in batch:
        # item['base_samples'] is (S, N)
        n = item['base_samples'].shape[1]
        if n > max_nodes:
            max_nodes = n
            
    # 2. Pad & Collect
    new_batch = {
        'base_samples': [],
        'int_samples': [],
        'target_row': [],
        'int_mask': [],
        'delta': [],
        'adj': [],
        'int_node_idx': []
    }
    
    for item in batch:
        # item is a chunk of S samples from one graph
        S, N = item['base_samples'].shape
        diff = max_nodes - N
        
        # Pad features: (S, N) -> (S, Max_N)
        # F.pad for 2D tensor (S, N): (pad_left, pad_right, pad_top, pad_bottom)
        # We only pad feature dim (last dim)
        pad_config_2d = (0, diff) # Pad right of last dim
        
        new_batch['base_samples'].append(F.pad(item['base_samples'], pad_config_2d))
        new_batch['int_samples'].append(F.pad(item['int_samples'], pad_config_2d))
        new_batch['target_row'].append(F.pad(item['target_row'], pad_config_2d))
        new_batch['delta'].append(F.pad(item['delta'], pad_config_2d))
        
        # int_mask is (S, N) because we expanded it in Dataset
        new_batch['int_mask'].append(F.pad(item['int_mask'], pad_config_2d))
        
        # int_node_idx is (S,) - No padding needed
        new_batch['int_node_idx'].append(item['int_node_idx'])
        
        # ADJ is (N, N) - Needs Padding + Expansion to (S, Max_N, Max_N)
        # Pad (N, N) -> (Max_N, Max_N)
        adj_padded = F.pad(item['adj'], (0, diff, 0, diff))
        # Expand to (S, Max_N, Max_N)
        adj_expanded = adj_padded.unsqueeze(0).expand(S, -1, -1)
        new_batch['adj'].append(adj_expanded)

    # 3. Concatenate (Flatten into one massive batch)
    collated = {}
    for k in new_batch:
        collated[k] = torch.cat(new_batch[k], dim=0)
            
    return collated
