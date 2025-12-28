import argparse
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from src.models.CausalTransformer import CausalTransformer
from src.data.SCMGenerator import SCMGenerator
from src.data.CausalDataset import CausalDataset
from src.data.collate import collate_fn_pad
from src.training.trainer import train_model 
from src.training.curriculum import CurriculumManager

def setup_ddp():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="ISD-CP Unified Training")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_vars", type=int, default=20)
    parser.add_argument("--max_vars", type=int, default=50)
    parser.add_argument("--edge_prob", type=float, default=None, help="Fixed edge probability (overrides curriculum)")
    parser.add_argument("--intervention_prob", type=float, default=0.5, help="Probability of intervening on a node")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 step to verify pipeline")
    
    args = parser.parse_args()
    
    local_rank = setup_ddp()
    is_master = (local_rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if is_master:
        print(f"--- ISD-CP Unified Training ---")
        print(f"Structure: Interleaved Tokens | Architecture: Hyper-Experts")
        print(f"Data: Twin World Variance Reduction")
        print(f"Device: {device}")

    # 1. Model
    # Max vars + buffer for embeddings
    model = CausalTransformer(num_nodes=args.max_vars + 5, d_model=128)
    model.to(device)
    
    if dist.is_initialized():
        print("Moving to DDP...")
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    # 2. Data & Curriculum
    curriculum = CurriculumManager(min_vars=args.min_vars, max_vars=args.max_vars)
    
    # 3. Training Loop (Delegate to Trainer/Curriculum loop)
    # Note: Our existing trainer.py 'train_model' is a simple loop.
    # We should refactor it slightly to accept the curriculum manager or run the loop here.
    # For V1 Unification, let's run a simple loop here.
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        # Update Curriculum Stats
        params = curriculum.get_current_params()
        if is_master:
            print(f"Epoch {epoch} | Level {curriculum.current_level} | Vars {params['max_vars']}")
        
        # Generator with current difficulty
        gen = SCMGenerator(
            num_nodes=params['max_vars'], 
            edge_prob=args.edge_prob if args.edge_prob is not None else params['density_max'],
            noise_scale=1.0,
            intervention_prob=args.intervention_prob
        )
        
        dataset = CausalDataset(
            gen, 
            num_nodes_range=(params['max_vars']-1, params['max_vars']),
            samples_per_graph=64
        )
        
        # Sampler for DDP
        sampler = DistributedSampler(dataset) if dist.is_initialized() else None
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn_pad, sampler=sampler)
        
        # Train 1 Epoch (which is infinite stream, so we define steps)
        steps_per_epoch = 50 if args.dry_run else 100
        
        model.train()
        total_loss = 0
        
        for i, batch in enumerate(dataloader):
            if i >= steps_per_epoch: break
            
            # Move to device
            base = batch['base_samples'].to(device)
            int_s = batch['int_samples'].to(device)
            target = batch['target_row'].to(device)
            mask = batch['int_mask'].to(device)
            idx = batch['int_node_idx'].to(device)
            
            # Forward
            deltas, logits, adj = model(base, int_s, target, mask, idx)
            
            # Loss (Simple MSE for Delta, BCE for DAG)
            delta_loss = torch.nn.functional.mse_loss(deltas, batch['delta'].to(device))
            
            loss = delta_loss 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if args.dry_run: break
            
        if is_master:
            avg_loss = total_loss / (i+1)
            print(f"Epoch {epoch} Loss: {avg_loss:.4f}")
            
        if args.dry_run:
            print("Dry Run Successful.")
            break
            
    cleanup_ddp()

if __name__ == "__main__":
    main()
