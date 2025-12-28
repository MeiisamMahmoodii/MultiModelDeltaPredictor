import torch
import time
import json
import os
from rich import print as rprint
from torch.utils.data import DataLoader
from src.data.SCMGenerator import SCMGenerator
from src.data.CausalDataset import CausalDataset
from src.data.collate import collate_fn_pad
from src.training.loss import causal_loss_fn

def compute_shd(pred_adj_logits, true_adj_matrix, threshold=0.0):
    pred_edges = (pred_adj_logits > threshold).float()
    true_edges = true_adj_matrix.float()
    diff = torch.abs(pred_edges - true_edges)
    shd = diff.sum(dim=(1, 2))
    return shd.mean().item()

def train_model(model, model_name, steps=3000, val_freq=100, lr=1e-4, num_nodes_range=(20, 50), edge_prob=0.3, intervention_prob=0.5, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    rprint(f"[bold white]Starting Training: {model_name} on {device}[/bold white]")
    
    gen = SCMGenerator()
    dataset = CausalDataset(gen, num_nodes_range=num_nodes_range, edge_prob=edge_prob, intervention_prob=intervention_prob)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn_pad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    history = {
        "step": [], "loss": [], "shd": [], 
        "loss_delta": [], "loss_dag": [], "loss_h": []
    }
    iter_loader = iter(dataloader)
    
    model.to(device)
    model.train()
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    try:
        for step in range(1, steps + 1):
            batch = next(iter_loader)
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            output = model(
                batch['base_samples'], 
                batch['int_samples'], 
                batch['target_row'],
                batch['int_mask'],
                int_node_idx=batch.get('int_node_idx', None)
            )
            
            if len(output) == 3:
                p_delta, p_adj_logits, _ = output
            else:
                p_delta, p_adj_logits = output
                
            loss, items = causal_loss_fn(p_delta, batch['delta'], p_adj_logits, batch['adj'])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % val_freq == 0 or step == 1:
                shd_val = compute_shd(p_adj_logits, batch['adj'])
                history['step'].append(step)
                history['loss'].append(loss.item())
                history['shd'].append(shd_val)
                history['loss_delta'].append(items['delta'])
                history['loss_dag'].append(items['dag'])
                history['loss_h'].append(items['h'])
                
                rprint(
                    f"Step {step} | Total: {loss.item():.2f} (Good < 50)| "
                    f"SHD: {shd_val:.1f} | "
                    f"Pred: {items['delta']:.2f} Good Range: < 5.0 Best:< 1.0| "
                    f"DAG: {items['dag']:.3f} Good Range: < 0.4| "
                    f"Cyc: {items['h']:.3f} Good Range: < 1.0 (approaching 0)"
                )
                
        torch.save(model.state_dict(), f"checkpoints/{model_name.replace(' ', '_')}_final.pt")
        with open(f"logs/{model_name.replace(' ', '_')}_log.json", 'w') as f:
            json.dump(history, f, indent=2)
            
        return history
    except KeyboardInterrupt:
        rprint("[yellow]Interrupted[/yellow]")
        return history
