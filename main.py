import argparse
import random
import csv
import os
from datetime import timedelta
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
        # Increase timeout to 30 mins for complex graph generation
        dist.init_process_group(backend, timeout=timedelta(minutes=30))
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


def get_validation_set(num_vars, device, edge_prob=0.2, intervention_prob=0.5, intervention_scale=1.0):
    """
    Generates a fixed validation set for a specific number of variables.
    Returns a DataLoader.
    """
    # Fixed parameters for validation to be a fair benchmark
    gen = SCMGenerator(
        num_nodes=num_vars, 
        edge_prob=edge_prob, 
        noise_scale=1.0, # Fixed noise for comparability
        intervention_prob=intervention_prob
    )
    # Update Generator's intervention values scale
    # Default is usually (-2, 2) roughly. If scale=5.0, we want (-10, 10).
    # SCMGenerator.intervention_values logic might need checking, but usually it's fixed.
    # Actually SCMGenerator line 186 in generator code: loop_values = [v * intervention_scale for v in self.intervention_values]
    # So we pass it to `generate_pipeline`? 
    # `CausalDataset` calls `generate_pipeline`.
    # `CausalDataset` accepts `intervention_scale_range`. 
    # If we want FIXED scale for validation, we should set range to (scale, scale).
    
    # Generate 16 fixed graphs for validation (Speed Optimization)
    dataset = CausalDataset(
        gen, 
        num_nodes_range=(num_vars, num_vars),
        samples_per_graph=32, # Reduced from 64
        infinite=False, 
        validation_graphs=16, # Reduced from 64
        intervention_prob=intervention_prob,
        intervention_scale_range=(intervention_scale, intervention_scale) # Fixed scale
    )
    return DataLoader(dataset, batch_size=32, collate_fn=collate_fn_pad)

def evaluate_loader(model, loader, device, description="Validating"):
    """
    Evaluates the model on a given dataloader.
    Returns a dictionary of aggregated metrics.
    """
    model.eval()
    total_metrics = {
        'mae': 0.0, 'f1': 0.0, 'tpr': 0.0, 'fdr': 0.0, 'n_batches': 0
    }
    
    with torch.no_grad():
        for batch in loader:
            base = batch['base_samples'].to(device)
            int_s = batch['int_samples'].to(device)
            target = batch['target_row'].to(device)
            mask = batch['int_mask'].to(device)
            idx = batch['int_node_idx'].to(device)
            
            # Forward
            deltas, logits, adj, _, _ = model(base, int_s, target, mask, idx)
            
            # Metrics
            total_metrics['mae'] += compute_mae(deltas, batch['delta'].to(device))
            total_metrics['f1'] += compute_f1(logits, batch['adj'].to(device))
            tpr, fdr = compute_tpr_fdr(logits, batch['adj'].to(device))
            total_metrics['tpr'] += tpr
            total_metrics['fdr'] += fdr
            total_metrics['n_batches'] += 1
            
    # Average
    n = max(1, total_metrics['n_batches'])
    return {
        'mae': total_metrics['mae'] / n,
        'f1': total_metrics['f1'] / n,
        'tpr': total_metrics['tpr'] / n,
        'fdr': total_metrics['fdr'] / n
    }

def main():
    parser = argparse.ArgumentParser(description="ISD-CP Unified Training")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4) # INCREASED 100x (Wake Up)
    parser.add_argument("--min_vars", type=int, default=20)
    parser.add_argument("--max_vars", type=int, default=50)
    parser.add_argument("--edge_prob", type=float, default=None, help="Fixed edge probability (overrides curriculum)")
    parser.add_argument("--intervention_prob", type=float, default=0.5, help="Probability of intervening on a node")
    parser.add_argument("--num_layers", type=int, default=16, help="Transformer Depth")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 step to verify pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--reuse_factor", type=int, default=1, help="Reuse each generated graph N times")
    parser.add_argument("--checkpoint_path", type=str, default="last_checkpoint.pt", help="Path to checkpoint")
    parser.add_argument("--grad_checkpoint", action="store_true", help="Enable gradient checkpointing (Saves Memory)")
    parser.add_argument("--lambda_dag", type=float, default=0.0, help="Weight for DAG structural loss")

    parser.add_argument("--lambda_h", type=float, default=0.0, help="Weight for Acyclicity loss")
    parser.add_argument("--lambda_sparse", type=float, default=0.0, help="Weight for Sparsity (L1) loss")
    
    # Ablation Flags
    parser.add_argument("--ablation_no_twin_world", action="store_true", help="Disable Twin World Variance Reduction")
    parser.add_argument("--ablation_dense_moe", action="store_true", help="Use Dense MLP instead of Hard MoE")
    parser.add_argument("--ablation_no_interleaved", action="store_true", help="Use standard additive encoding")
    parser.add_argument("--ablation_no_dag", action="store_true", help="Disable DAG Head")
    parser.add_argument("--ablation_no_physics", action="store_true", help="Disable Physics Head")
    
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
        num_layers=args.num_layers,
        grad_checkpoint=args.grad_checkpoint,
        ablation_dense=args.ablation_dense_moe,
        ablation_no_interleaved=args.ablation_no_interleaved,
        ablation_no_dag=args.ablation_no_dag,
        ablation_no_physics=args.ablation_no_physics
    )
    model.to(device)
    
    if is_master:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: {total_params:,}")
    
    if dist.is_initialized():
        print("Moving to DDP...")
        # find_unused_parameters=True is required for Hard MoE (sparse activation).
        # We must solve OOM via Batch Size / Gradient Checkpointing.
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
        
        # Handle state dict for DDP (Fixing key mismatch)
        state_dict = checkpoint['model_state_dict']
        
        # Check if model is wrapped in DDP (has 'module.')
        is_ddp_model = hasattr(model, 'module')
        
        # Check if checkpoint keys have 'module.'
        ckpt_has_module = list(state_dict.keys())[0].startswith('module.')
        
        if is_ddp_model and not ckpt_has_module:
            # Add 'module.' prefix
            if is_master: print("Adapting checkpoint (Single-GPU) -> Model (DDP)...")
            new_state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            state_dict = new_state_dict
        elif not is_ddp_model and ckpt_has_module:
            # Strip 'module.' prefix
            if is_master: print("Adapting checkpoint (DDP) -> Model (Single-GPU)...")
            new_state_dict = {k[7:]: v for k, v in state_dict.items()}
            state_dict = new_state_dict
            
        # Phase 5 Transition: Checkpoint might be missing DAG Head keys.
        # We use strict=False to allow loading partial weights (Physics) while initializing DAG Head randomly.
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if is_master and len(missing_keys) > 0:
            print(f"Warning: Missing keys in checkpoint (Expected for Phase 5 Transition): {missing_keys}")
        
        # Optimizer Load: Might fail if params changed.
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                # Force verbose off to avoid warnings from resumed state
                if hasattr(scheduler, 'verbose'): scheduler.verbose = False
        except Exception as e:
            if is_master:
                print(f"Warning: Could not load optimizer state (New Parameters Added?): {e}")
                print("Starting with FRESH Optimizer for this Phase.")
        curriculum.load_state_dict(checkpoint['curriculum_state_dict'])
        curriculum.load_state_dict(checkpoint['curriculum_state_dict'])
        
        # HOTFIX: Reset Router to break expert collapse from previous phase
        if is_master:
            print("Re-initializing MoE Router to break collapse...")
        with torch.no_grad():
            # Access router via module if wrapped
            if hasattr(model, 'module'):
                router = model.module.moe.router
            else:
                router = model.moe.router
            
            router.weight.normal_(0, 0.02)
            router.bias.zero_()
            
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
            # Use current density (or fixed edge_prob) 
            # Match Training Intervention Probability
            curr_edge_prob = args.edge_prob if args.edge_prob is not None else params['density_max']
            val_loader = get_validation_set(
                params['max_vars'], 
                device, 
                edge_prob=curr_edge_prob,
                intervention_prob=args.intervention_prob, # FIXED: Match training difficulty
                intervention_scale=params['intervention_range']
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
            reuse_factor=args.reuse_factor,
            use_twin_world=not args.ablation_no_twin_world,
            intervention_scale_range=(1.0, 50.0) # Extreme Physics Training: Train on small (1.0) to massive (50.0) interventions
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
            "tpr": 0.0, "fdr": 0.0, "aux_moe": 0.0
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
            # Forward (Phase 4: Returns deltas, logits, adj, mcm_out, aux_loss)
            deltas, logits, adj, _, aux_loss = model(base, int_s, target, mask, idx)
            
            # Loss (Full Causal Loss: Delta + DAG + Acyclicity)
            loss, items = causal_loss_fn(
                deltas, 
                batch['delta'].to(device), 
                logits, 
                batch['adj'].to(device),
                lambda_dag=args.lambda_dag,
                lambda_h=args.lambda_h,
                lambda_l1=args.lambda_sparse
            ) 
            
            # Add Load Balancing Loss
            loss += 0.1 * aux_loss # Lambda Aux for Load Balancing
            items['aux_moe'] = aux_loss.item()
            
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
                
            if is_master:
                # Retrieve MoE Metrics (Handle DDP wrapper)
                if hasattr(model, 'module'):
                    moe_metrics = model.module.moe.get_expert_metrics()
                else:
                    moe_metrics = model.moe.get_expert_metrics()
                
                avg_loss = total_loss / (i + 1)
                avg_shd = total_metrics['shd'] / (i + 1)
                avg_f1 = total_metrics['f1'] / (i + 1)
                avg_mae = total_metrics['mae'] / (i + 1)
                avg_delta = total_metrics['delta'] / (i + 1)
                
                metric_str = f"L: {avg_loss:.1f} | Δ: {avg_delta:.2f} | MAE: {avg_mae:.2f} | SHD:{avg_shd:.1f} | Ent: {moe_metrics['entropy']:.2f} | Gini: {moe_metrics['gini']:.2f}"
                
                if RICH_AVAILABLE and progress is not None:
                    progress.update(task_id, advance=1, metrics=metric_str)
                else:
                    # ASCII logic simplified
                    print(f"\rStep {i+1} | {metric_str}", end='', flush=True)

            if args.dry_run: break
            
        # --- END OF EPOCH BLOCK (Outside Loop) ---
        if is_master:
            if progress: progress.stop()
            if not RICH_AVAILABLE: print(flush=True)
            
        # Synchronize before validation to prevent timeouts
        if dist.is_initialized():
            dist.barrier()
            
        # --- Validation Loop (Fixed Set & Benchmarks) ---
        # 1. Evaluate Current Level (All Ranks participate to sync Curriculum)
        val_metrics = evaluate_loader(model, val_loader, device)
        val_mae = val_metrics['mae']
        val_f1 = val_metrics['f1']
        
        if is_master:
            print(f"Val Level {curriculum.current_level} | MAE: {val_mae:.3f} | F1: {val_f1:.3f} | TPR: {val_metrics['tpr']:.2f} | FDR: {val_metrics['fdr']:.2f}")
            
            # 2. Cross-Difficulty Benchmarks (Master Only)
            # "Novel Solution: Cross-Difficulty Validation"
            benchmarks = curriculum.get_benchmark_params()
            print("--- Cross-Difficulty Benchmarks ---")
            for level_name, b_params in benchmarks.items():
                # Generate ephemeral loader for benchmark
                # Use max density for robust testing
                b_loader = get_validation_set(
                    b_params['max_vars'],
                    device,
                    edge_prob=b_params['density_max'], 
                    intervention_prob=args.intervention_prob,
                    intervention_scale=b_params['intervention_range']
                )
                b_metrics = evaluate_loader(model, b_loader, device, description=level_name)
                print(f"[{level_name.upper()}] MAE: {b_metrics['mae']:.3f} | F1: {b_metrics['f1']:.3f}")
            print("-----------------------------------")

        val_tpr = val_metrics['tpr']
        val_fdr = val_metrics['fdr']
        
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
            # Re-fetch metrics for summary
            if hasattr(model, 'module'):
                moe_metrics = model.module.moe.get_expert_metrics()
            else:
                moe_metrics = model.moe.get_expert_metrics()

            if RICH_AVAILABLE:
                table = Table(title=f"Epoch {epoch} Summary | Level {curriculum.current_level}")
                table.add_column("Metric", style="cyan", no_wrap=True)
                table.add_column("Train", style="magenta")
                table.add_column("Val (Fixed)", style="green")
                
                table.add_row("Total Loss", f"{avg_loss:.4f}", "-")
                table.add_row("MAE (L1)", f"{total_metrics['mae']/(i+1):.4f}", f"{val_mae:.4f}")
                table.add_row("SHD", f"{avg_shd:.2f}", "-")
                table.add_row("Expert Entropy", f"{moe_metrics['entropy']:.4f}", "-")
                table.add_row("Expert Gini", f"{moe_metrics['gini']:.4f}", "-")
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
