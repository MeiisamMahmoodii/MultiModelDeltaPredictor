import subprocess
import csv
import os
import pandas as pd

# Define Ablations
# Name: [List of flags]
ABLATIONS = {
    "Full Model": [],
    "No Interleaved (Additive)": ["--ablation_no_interleaved"],
    "No Physics Head": ["--ablation_no_physics"],
    "No DAG Head": ["--ablation_no_dag"],
    "Dense MLP (No MoE)": ["--ablation_dense_moe"],
    "No Twin World": ["--ablation_no_twin_world"]
}

def run_ablation_suite(epochs=5, output_file='ablation_results.csv', dry_run=False):
    results = []
    
    print(f"--- Starting Ablation Suite ---")
    print(f"Epochs per run: {epochs}")
    
    for name, flags in ABLATIONS.items():
        print(f"\n[Ablation] {name}")
        
        # Command construction
        import sys
        cmd = [
            sys.executable, "main.py",
            "--epochs", str(epochs),
            "--batch_size", "16", # Keep small for speed
            "--min_vars", "20",   # Standard size
            "--max_vars", "20",   # Fixed size for clean comparison
            "--intervention_prob", "0.2",
            "--lambda_dag", "1.0", # Ensure heads are active
            "--lambda_sparse", "0.1"
        ] + flags
        
        if dry_run:
            cmd.append("--dry_run")
            
        print(f"  Command: {' '.join(cmd)}")
        
        try:
            # Run the command
            # Capture output? Or just let it print?
            # We need to capture the FINAL metrics.
            # main.py saves to `training_log.csv`. 
            # We should probably clear training_log.csv before each run or read the last line.
            
            # Delete old log if exists to avoid confusion
            if os.path.exists("training_log.csv"):
                os.remove("training_log.csv")
                
            subprocess.run(cmd, check=True)
            
            # Read Results
            if os.path.exists("training_log.csv"):
                df = pd.read_csv("training_log.csv")
                if not df.empty:
                    last_row = df.iloc[-1]
                    results.append({
                        "Ablation": name,
                        "Val_MAE": last_row.get("Val_MAE", -1),
                        "Val_F1": last_row.get("Val_F1", -1),
                        "Val_TPR": last_row.get("Val_TPR", -1),
                        "Train_Loss": last_row.get("Train_Loss", -1)
                    })
                else:
                    print("  Warning: training_log.csv is empty.")
            else:
                 print("  Warning: training_log.csv not found.")
                 
        except subprocess.CalledProcessError as e:
            print(f"  Failed: {e}")
            
        if dry_run: break

    # Save Aggregate Results
    if results:
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"\nSaved ablation results to {output_file}")
    else:
        print("\nNo results collected.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    
    run_ablation_suite(epochs=args.epochs, dry_run=args.dry_run)
