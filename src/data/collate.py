import torch
import torch.nn.functional as F

def collate_fn_pad(batch):
    """
    Pads batch of variable-size graphs to the max size in the batch.
    """
    # Keys: base_samples, int_samples, target_row, int_mask, delta, adj, int_node_idx
    
    # 1. Find Max Nodes in this batch
    # base_samples shape: (Samples, Nodes) ?? No, in dataset it is (Samples, Nodes)
    # Wait, check CausalDataset yield:
    # "base_samples": base_tensor (100, N)
    max_nodes = 0
    for item in batch:
        n = item['base_samples'].shape[1]
        if n > max_nodes:
            max_nodes = n
            
    # 2. Pad Items
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
        n = item['base_samples'].shape[1]
        diff = max_nodes - n
        
        # Pad (Last dim, 2nd last dim...)
        # F.pad expects (left, right, top, bottom, ...)
        
        # base_samples: (S, N) -> Pad dim 1 (right)
        new_batch['base_samples'].append(F.pad(item['base_samples'], (0, diff)))
        
        # int_samples: (S, N) -> Pad dim 1
        new_batch['int_samples'].append(F.pad(item['int_samples'], (0, diff)))
        
        # target_row: (N) -> Pad dim 0
        new_batch['target_row'].append(F.pad(item['target_row'], (0, diff)))
        
        # int_mask: (N) -> Pad dim 0
        new_batch['int_mask'].append(F.pad(item['int_mask'], (0, diff)))
        
        # delta: (N) -> Pad dim 0
        new_batch['delta'].append(F.pad(item['delta'], (0, diff)))
        
        # adj: (N, N) -> Pad dim 0/1 (right, bottom)
        new_batch['adj'].append(F.pad(item['adj'], (0, diff, 0, diff)))
        
        # int_node_idx is scalar
        new_batch['int_node_idx'].append(item['int_node_idx'])

    # 3. Stack
    collated = {}
    for k in new_batch:
        if k == 'int_node_idx':
            collated[k] = torch.stack(new_batch[k])
        else:
            collated[k] = torch.stack(new_batch[k])
            
    return collated
