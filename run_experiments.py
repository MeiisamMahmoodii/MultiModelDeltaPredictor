import torch
from src.models.ModelA_Baseline import ModelA_Baseline
from src.models.ModelB_Experts import ModelB_Experts
from src.models.ModelC_Sparsity import ModelC_Sparsity
from src.models.ModelD_Masked import ModelD_Masked
from src.models.ModelE_HyperNet import ModelE_HyperNet
from src.models.ModelF_Unified import ModelF_Unified
from src.training.trainer import train_model

def main():
    # --- CONFIGURATION (Modify Here) ---
    CONFIG = {
        "num_nodes_range": (20, 20),  # Min/Max nodes in generated graphs
        "edge_prob": 0.3,             # Probability of edge connection
        "intervention_prob": 0.5,     # % of nodes to intervene on
        "d_model": 128,               # Transformer embedding size
        "nhead": 8,                   # Number of attention heads
        "num_layers": 8,              # Number of transformer layers
        "steps": 5000,                # Training steps
        "lr": 1e-4,                   # Learning rate
        "model_capacity_nodes": 20,   # Max nodes model can handle (embeddings)
    }
    
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Running Experiments on: {device}")
    print(f"Config: {CONFIG}")
    
    n_nodes = CONFIG['model_capacity_nodes']
    d_model = CONFIG['d_model']
    nhead = CONFIG['nhead']
    n_layers = CONFIG['num_layers']
    steps = CONFIG['steps']
    
    # helper for clean calling
    def run_training(model, name):
        train_model(
            model, name, 
            steps=steps, 
            lr=CONFIG['lr'],
            num_nodes_range=CONFIG['num_nodes_range'],
            edge_prob=CONFIG['edge_prob'],
            intervention_prob=CONFIG['intervention_prob'],
            device=device
        )

    # 1. Train Model A
    print("\n--- Training Model A ---")
    model_a = ModelA_Baseline(num_nodes=n_nodes, d_model=d_model, nhead=nhead, num_layers=n_layers)
    run_training(model_a, "Model A Baseline")
    
    # 2. Train Model B
    print("\n--- Training Model B ---")
    model_b = ModelB_Experts(num_nodes=n_nodes, d_model=d_model, nhead=nhead, num_layers=n_layers)
    run_training(model_b, "Model B Experts")
    
    # 3. Train Model C
    print("\n--- Training Model C ---")
    model_c = ModelC_Sparsity(num_nodes=n_nodes, d_model=d_model, nhead=nhead, num_layers=n_layers)
    run_training(model_c, "Model C Sparsity")
    
    # 4. Train Model D
    print("\n--- Training Model D ---")
    model_d = ModelD_Masked(num_nodes=n_nodes, d_model=d_model, nhead=nhead, num_layers=n_layers)
    run_training(model_d, "Model D Masked")
    
    # 5. Train Model E
    print("\n--- Training Model E ---")
    # Model E constructor might vary slightly (check if it accepts num_layers)
    # Checking ModelE signature... it has hardcoded backbone inside usually.
    # Let's verify Model E signature. Assuming it needs updates to accept nhead/layers too.
    # For now, pass what matched previous calls, but ideally we update all models to match config.
    # ModelE_HyperNet(num_nodes, d_model) was the old call.
    # Let's inspect ModelE source later or assume it takes d_model.
    # The snippet previously showed: ModelE_HyperNet(num_nodes, d_model)
    model_e = ModelE_HyperNet(num_nodes=n_nodes, d_model=d_model) 
    run_training(model_e, "Model E HyperNet")
    
    # 6. Train Model F (Unified)
    print("\n--- Training Model F (Unified) ---")
    model_f = ModelF_Unified(num_nodes=n_nodes, d_model=d_model, nhead=nhead, num_layers=n_layers)
    run_training(model_f, "Model F Unified")
    
    print("\nAll Experiments Completed. Check logs/ folder for results.")

if __name__ == "__main__":
    main()
