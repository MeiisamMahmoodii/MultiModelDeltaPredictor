import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.CausalTransformer import CausalTransformer
from src.data.SCMGenerator import SCMGenerator
from src.data.CausalDataset import CausalDataset
from src.data.collate import collate_fn_pad
from torch.utils.data import DataLoader

def setup_plotting():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12

def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint['args']
    
    # Reconstruct Model
    # Note: args is a Namespace object
    model = CausalTransformer(
        num_nodes=args.max_vars + 5, # Buffer
        d_model=512, # Hardcoded in main.py Phase 4
        num_layers=args.num_layers,
        grad_checkpoint=False # No need for checkpointing during inference
    )
    
    # Handle DDP prefix if present
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model, args

def generate_validation_data(args, device, num_samples=512):
    print(f"Generating {num_samples} validation samples...")
    # Use the max_vars from training for the hardest test
    n_vars = args.max_vars
    
    gen = SCMGenerator(
        num_nodes=n_vars,
        edge_prob=0.25, # High density
        noise_scale=1.0,
        intervention_prob=0.5 # Mix of interventions
    )
    
    dataset = CausalDataset(
        gen,
        num_nodes_range=(n_vars, n_vars),
        samples_per_graph=32,
        infinite=False,
        validation_graphs=max(1, num_samples // 32),
        intervention_prob=0.1
    )
    
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn_pad)
    return loader

def run_inference(model, loader, device):
    all_preds = []
    all_trues = []
    all_values = [] # Store raw values for context
    
    print("Running Inference...")
    with torch.no_grad():
        for batch in loader:
            base = batch['base_samples'].to(device)
            int_s = batch['int_samples'].to(device)
            target = batch['target_row'].to(device)
            mask = batch['int_mask'].to(device)
            idx = batch['int_node_idx'].to(device)
            true_delta = batch['delta'].to(device)
            
            # Forward
            pred_delta, _, _, _ = model(base, int_s, target, mask, idx)
            
            all_preds.append(pred_delta.cpu().numpy())
            all_trues.append(true_delta.cpu().numpy())
            all_values.append(target.cpu().numpy())
            
    return np.concatenate(all_preds), np.concatenate(all_trues), np.concatenate(all_values)

def plot_results(preds, trues, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten
    preds_flat = preds.flatten()
    trues_flat = trues.flatten()
    
    # 1. Parity Plot (Scatter)
    plt.figure(figsize=(10, 10))
    # Downsample for scatter plot if too large
    if len(preds_flat) > 10000:
        indices = np.random.choice(len(preds_flat), 10000, replace=False)
        p_plot = preds_flat[indices]
        t_plot = trues_flat[indices]
    else:
        p_plot = preds_flat
        t_plot = trues_flat
        
    plt.scatter(t_plot, p_plot, alpha=0.3, s=10, color='blue', label='Predictions')
    
    # Ideal line
    min_val = min(t_plot.min(), p_plot.min())
    max_val = max(t_plot.max(), p_plot.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (x=y)')
    
    plt.title("Parity Plot: True Delta vs Predicted Delta")
    plt.xlabel("True Delta")
    plt.ylabel("Predicted Delta")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "parity_plot.png"))
    plt.close()
    
    # 2. Error Histogram
    errors = preds_flat - trues_flat
    plt.figure(figsize=(12, 6))
    sns.histplot(errors, bins=100, kde=True, color='purple')
    plt.title(f"Error Distribution (Mean: {np.mean(errors):.2f}, Std: {np.std(errors):.2f})")
    plt.xlabel("Prediction Error (Pred - True)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "error_dist.png"))
    plt.close()
    
    # 3. Residual Plot
    plt.figure(figsize=(12, 6))
    if len(preds_flat) > 5000:
         indices = np.random.choice(len(preds_flat), 5000, replace=False)
         p_plot = preds_flat[indices]
         e_plot = errors[indices]
    else:
         p_plot = preds_flat
         e_plot = errors
         
    plt.scatter(p_plot, e_plot, alpha=0.3, s=10, color='green')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals vs Predicted Value")
    plt.xlabel("Predicted Delta")
    plt.ylabel("Residual (Error)")
    plt.ylim(-50, 50) # Zoom in on the main cluster
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "residuals.png"))
    plt.close()

    print(f"Plots saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="last_checkpoint.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=1024)
    args = parser.parse_args()
    
    setup_plotting()
    device = torch.device(args.device)
    
    try:
        model, train_args = load_model(args.checkpoint, device)
        loader = generate_validation_data(train_args, device, args.samples)
        preds, trues, values = run_inference(model, loader, device)
        plot_results(preds, trues)
        
        # Metrics Printout
        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues)**2)
        print(f"\n--- Validation Metrics ---")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"Data Range: {trues.min():.2f} to {trues.max():.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
