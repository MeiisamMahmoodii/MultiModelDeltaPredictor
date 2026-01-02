import argparse
import random
import csv
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from src.models.CausalTransformer import CausalTransformer
from src.data.SCMGenerator import SCMGenerator
from src.data.CausalDataset import CausalDataset
from src.data.collate import collate_fn_pad
from src.training.loss import causal_loss_fn
from src.training.curriculum import CurriculumManager
from src.training.metrics import compute_shd, compute_f1, compute_mae, compute_tpr_fdr
try:
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.table import Table
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

def setup_ddp():
    if "LOCAL_RANK" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        local_rank = int(os.environ["LOCAL_RANK"])
        import numpy as np
        np.random.seed(local_rank) # Ensure different data per rank
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        # MPS doesn't support set_device like CUDA, handled by device object later
        return local_rank
    return 0

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_validation_set(num_vars, device, edge_prob=0.2, intervention_prob=0.5):
    """
    Generates a fixed validation set for a specific number of variables.
    Returns a DataLoader.
    """
    # Fixed parameters for validation to be a fair benchmark
    gen = SCMGenerator(
        num_nodes=num_vars, 
        edge_prob=edge_prob, 
        noise_scale=1.0,
        intervention_prob=intervention_prob
    )
    # Generate 32 fixed graphs for validation
    dataset = CausalDataset(
        gen, 
        num_nodes_range=(num_vars, num_vars),
        samples_per_graph=64,
        infinite=False, # Important: Fixed size
        validation_graphs=64
    )
    return DataLoader(dataset, batch_size=32, collate_fn=collate_fn_pad)

def main():
    parser = argparse.ArgumentParser(description="ISD-CP Unified Training")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--min_vars", type=int, default=20)
    parser.add_argument("--max_vars", type=int, default=50)
    parser.add_argument("--edge_prob", type=float, default=None, help="Fixed edge probability (overrides curriculum)")
    parser.add_argument("--intervention_prob", type=float, default=0.5, help="Probability of intervening on a node")
    parser.add_argument("--num_layers", type=int, default=16, help="Transformer Depth")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 step to verify pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--reuse_factor", type=int, default=1, help="Reuse each generated graph N times")
    parser.add_argument("--checkpoint_path", type=str, default="last_checkpoint.pt", help="Path to checkpoint")
    
    args = parser.parse_args()
    
    local_rank = setup_ddp()
    is_master = (local_rank == 0)
    
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = f"cuda:{local_rank}"
    elif torch.backends.mps.is_available():
        device_name = "mps"
        
    device = torch.device(device_name)
    
    if is_master:
        print(f"--- ISD-CP Unified Training ---")
        print(f"Structure: Interleaved Tokens | Architecture: Hyper-Experts")
        print(f"Data: Twin World Variance Reduction")
        print(f"Device: {device}")
        
        # Create checkpoints directory
        os.makedirs("checkpoints", exist_ok=True)

    # 1. Model
    # Max vars + buffer for embeddings
    # Increased d_model to 512 for Phase 3 "Physics-Native" Capacity
    model = CausalTransformer(
        num_nodes=args.max_vars + 5, 
        d_model=512,
        num_layers=args.num_layers
    )
    model.to(device)
    
    if is_master:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: {total_params:,}")
    
    if dist.is_initialized():
        print("Moving to DDP...")
        # find_unused_parameters=True is required because:
        # 1. Curriculum means we don't use all experts in early stages.
        # 2. Some heads might be skipped conditionally.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # 2. Data & Curriculum
    curriculum = CurriculumManager(min_vars=args.min_vars, max_vars=args.max_vars)
    
    # 3. Training Loop (Delegate to Trainer/Curriculum loop)
    # Note: Our existing trainer.py 'train_model' is a simple loop.
    # We should refactor it slightly to accept the curriculum manager or run the loop here.
    # For V1 Unification, let's run a simple loop here.
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Scheduler: Cosine Annealing with Warm Restarts
    # Restart every 50 epochs, doubling the period each time (T_mult=2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-8
    )
    
    start_epoch = 0
    
    # Validation Loader (Dynamic)
    val_loader = None
    current_val_vars = -1
    
    start_epoch = 0
    
    # Resume Logic
    if args.resume and os.path.exists(args.checkpoint_path):
        if is_master:
            print(f"Resuming from {args.checkpoint_path}...")
        # Map location is important for DDP
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(args.checkpoint_path, map_location=map_location, weights_only=False)
        
        # Handle state dict for DDP (remove 'module.' prefix if needed or add it)
        # However, DDP wraps model in 'module.', so direct load normally works if saved from DDP.
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # Force verbose off to avoid warnings from resumed state
            if hasattr(scheduler, 'verbose'): scheduler.verbose = False
        curriculum.load_state_dict(checkpoint['curriculum_state_dict'])
        curriculum.load_state_dict(checkpoint['curriculum_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if is_master:
            print(f"Resumed at Epoch {start_epoch} | Level {curriculum.current_level}")
    
    for epoch in range(start_epoch, args.epochs):
        # Update Curriculum Stats
        params = curriculum.get_current_params()
        
        # Check if we need to regenerate validation set (Level Changed or First Run)
        if params['max_vars'] != current_val_vars:
            if is_master: 
                print(f"Generating new Validation Set for {params['max_vars']} vars...")
            # We generate Val Set on all ranks to avoid broadcasting complexity for now 
            # (RNG seeded by local_rank, so each rank validates on its own slice)
            # Use current density (or fixed edge_prob) + current int_prob
            curr_edge_prob = args.edge_prob if args.edge_prob is not None else params['density_max']
            val_loader = get_validation_set(
                params['max_vars'], 
                device, 
                edge_prob=curr_edge_prob,
                intervention_prob=args.intervention_prob
            )
            current_val_vars = params['max_vars']
            
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
            samples_per_graph=64,
            reuse_factor=args.reuse_factor
        )
        
        # No DistributedSampler for IterableDataset
        # Each rank has its own process and generator state.
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn_pad, sampler=None)
        
        # Train 1 Epoch (which is infinite stream, so we define steps)
        # "Infinite" Dataset: We define an epoch as 2000 steps (~64k samples)
        steps_per_epoch = 50 if args.dry_run else 2000
        
        model.train()
        total_loss = 0
        total_metrics = {
            "delta": 0.0, "dag": 0.0, "h": 0.0,
            "shd": 0.0, "f1": 0.0, "mae": 0.0,
            "tpr": 0.0, "fdr": 0.0
        }
        
        # Progress Bar (Only on Master)
        progress = None
        if is_master:
            if RICH_AVAILABLE:
                progress = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeRemainingColumn(),
                    TextColumn("[magenta]{task.fields[metrics]}")
                )
                task_id = progress.add_task(
                    f"[cyan]Epoch {epoch}", 
                    total=steps_per_epoch, 
                    metrics="Loss: ..."
                )
                progress.start()
            else:
                # ASCII Header
                print(f"Epoch {epoch} Started", flush=True)
        
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
            
            # Loss (Full Causal Loss: Delta + DAG + Acyclicity)
            loss, items = causal_loss_fn(
                deltas, 
                batch['delta'].to(device), 
                logits, 
                batch['adj'].to(device)
            ) 
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1) # Logic fix for instability
            optimizer.step()
            
            # Additional Metrics
            with torch.no_grad():
                shd = compute_shd(logits, batch['adj'].to(device))
                f1 = compute_f1(logits, batch['adj'].to(device))
                mae = compute_mae(deltas, batch['delta'].to(device))
                tpr, fdr = compute_tpr_fdr(logits, batch['adj'].to(device))
            
            # Track Metrics
            total_loss += loss.item()
            for k, v in items.items():
                total_metrics[k] += v
            total_metrics['shd'] += shd
            total_metrics['f1'] += f1
            total_metrics['mae'] += mae
            total_metrics['tpr'] += tpr
            total_metrics['fdr'] += fdr
                
            # Update Progress
            if is_master:
                avg_loss = total_loss / (i + 1)
                avg_shd = total_metrics['shd'] / (i + 1)
                avg_f1 = total_metrics['f1'] / (i + 1)
                avg_mae = total_metrics['mae'] / (i + 1)
                
                avg_delta = total_metrics['delta'] / (i + 1)
                metric_str = f"L: {avg_loss:.1f} | Δ: {avg_delta:.2f} | MAE: {avg_mae:.2f} | SHD:{avg_shd:.1f} | F1: {avg_f1:.2f} | TPR: {tpr:.2f} | FDR: {fdr:.2f}"
                
                if RICH_AVAILABLE and progress is not None:
                    progress.update(task_id, advance=1, metrics=metric_str)
                else:
                    # ASCII Bar Logic
                    percent = (i+1) / steps_per_epoch
                    bar_len = 30
                    filled_len = int(bar_len * percent)
                    bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len - 1)
                    if percent == 1.0: bar = '=' * bar_len
                    
                    # \r overwrites the line
                    print(f"\r[{bar}] {int(percent*100)}% | step: {i+1} | {metric_str}", end='', flush=True)
            
            if args.dry_run: break
            
        # --- END OF EPOCH BLOCK (Outside Loop) ---
        if is_master:
            if progress: progress.stop()
            if not RICH_AVAILABLE: print(flush=True) # Newline after ASCII bar
            
        # --- Validation Loop (Fixed Set) ---
        model.eval()
        val_mae_sum = 0.0
        val_f1_sum = 0.0
        val_tpr_sum = 0.0
        val_fdr_sum = 0.0
        val_batches = 0
        
        # Validation Progress Bar
        val_progress = None
        if is_master:
            print(f"Validating on Fixed Set...", flush=True)
            if RICH_AVAILABLE:
                val_progress = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeRemainingColumn(),
                    TextColumn("[magenta]{task.fields[metrics]}")
                )
                # Fixed val set: 32 graphs * 64 samples / 32 batch_size = 64 batches
                val_steps = 64
                val_task = val_progress.add_task("[green]Validating", total=val_steps, metrics="MAE: ...")
                val_progress.start()

        with torch.no_grad():
            for i, val_batch in enumerate(val_loader):
                base = val_batch['base_samples'].to(device)
                int_s = val_batch['int_samples'].to(device)
                target = val_batch['target_row'].to(device)
                mask = val_batch['int_mask'].to(device)
                idx = val_batch['int_node_idx'].to(device)
                
                deltas, logits, adj = model(base, int_s, target, mask, idx)
                
                # Metrics
                batch_mae = compute_mae(deltas, val_batch['delta'].to(device))
                batch_f1 = compute_f1(logits, val_batch['adj'].to(device))
                batch_tpr, batch_fdr = compute_tpr_fdr(logits, val_batch['adj'].to(device))
                
                val_mae_sum += batch_mae
                val_f1_sum += batch_f1
                val_tpr_sum += batch_tpr
                val_fdr_sum += batch_fdr
                val_batches += 1
                
                if is_master and val_progress:
                     val_progress.update(val_task, advance=1, metrics=f"MAE: {val_mae_sum/val_batches:.3f}")

        if is_master and val_progress:
            val_progress.stop()
                
        val_mae = val_mae_sum / max(1, val_batches)
        val_f1 = val_f1_sum / max(1, val_batches)
        val_tpr = val_tpr_sum / max(1, val_batches)
        val_fdr = val_fdr_sum / max(1, val_batches)
        
        # Step Scheduler (Cosine uses epoch, not val metric)
        scheduler.step(epoch + i / steps_per_epoch) # Update with partial epoch for smoother cosine
        
        # Calculate Epoch Metrics (Training Avg)
        i = max(1, i) # Avoid div by zero if loop didn't run
        avg_loss = total_loss / (max(1, i+1))
        # avg_mae = total_metrics['mae'] / (max(1, i+1)) # Using Validation MAE for curriculum now
        avg_shd = total_metrics['shd'] / (max(1, i+1)) 
        
        # 1. Update Curriculum (using VALIDATION Scores)
        leveled_up, reset_lr = curriculum.update(val_mae, val_f1)
        
        # 2. Print Summary (Master Only)
        if is_master:
            if RICH_AVAILABLE:
                table = Table(title=f"Epoch {epoch} Summary | Level {curriculum.current_level}")
                table.add_column("Metric", style="cyan", no_wrap=True)
                table.add_column("Train", style="magenta")
                table.add_column("Val (Fixed)", style="green")
                
                table.add_row("Total Loss", f"{avg_loss:.4f}", "-")
                table.add_row("MAE (L1)", f"{total_metrics['mae']/(i+1):.4f}", f"{val_mae:.4f}")
                table.add_row("SHD", f"{avg_shd:.2f}", "-")
                table.add_row("F1 Score", f"{total_metrics['f1']/(i+1):.4f}", f"{val_f1:.4f}")
                table.add_row("TPR", f"{total_metrics['tpr']/(i+1):.4f}", f"{val_tpr:.4f}")
                table.add_row("LR", f"{optimizer.param_groups[0]['lr']:.2e}", "")
                
                rprint(table)
                if leveled_up:
                    rprint(f"[bold yellow]*** LEVEL UP! Advanced to Level {curriculum.current_level} ***[/bold yellow]")
            else:
                print(f"Ep {epoch}|L:{avg_loss:.2f}|TrMAE:{total_metrics['mae']/(i+1):.2f}|ValMAE:{val_mae:.2f}|LR:{optimizer.param_groups[0]['lr']:.2e}")
                if leveled_up:
                    print(f"*** LEVEL UP! Advanced to Level {curriculum.current_level} ***")
            
            # 2b. CSV Logging
            log_file = "training_log.csv"
            file_exists = os.path.isfile(log_file)
            with open(log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                header = ["Epoch", "Level", "LR", 
                          "Train_Loss", "Train_Delta", "Train_DAG", "Train_H", 
                          "Train_MAE", "Train_SHD", "Train_F1", "Train_TPR", "Train_FDR", 
                          "Val_MAE", "Val_F1", "Val_TPR", "Val_FDR"]
                if not file_exists:
                    writer.writerow(header)
                
                train_iters = max(1, i+1) # total training iterations this epoch
                writer.writerow([
                    epoch, 
                    curriculum.current_level,
                    optimizer.param_groups[0]['lr'],
                    f"{avg_loss:.4f}",
                    f"{total_metrics['delta']/train_iters:.4f}",
                    f"{total_metrics['dag']/train_iters:.4f}",
                    f"{total_metrics['h']/train_iters:.4f}",
                    f"{total_metrics['mae']/train_iters:.4f}",
                    f"{avg_shd:.4f}",
                    f"{total_metrics['f1']/train_iters:.4f}",
                    f"{total_metrics['tpr']/train_iters:.4f}",
                    f"{total_metrics['fdr']/train_iters:.4f}",
                    f"{val_mae:.4f}",
                    f"{val_f1:.4f}",
                    f"{val_tpr:.4f}",
                    f"{val_fdr:.4f}"
                ])
        
        # 3. Save Checkpoint (Master Only)
        if is_master:
            # 1. Save Resume Checkpoint
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'curriculum_state_dict': curriculum.state_dict(),
                'args': args
            }
            torch.save(checkpoint_data, args.checkpoint_path)
            
            # 2. Save Historical Snapshot
            snapshot_path = f"checkpoints/checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint_data, snapshot_path)
            
        if args.dry_run:
            print("Dry Run Successful.")
            break
            
    cleanup_ddp()

if __name__ == "__main__":
    main()
