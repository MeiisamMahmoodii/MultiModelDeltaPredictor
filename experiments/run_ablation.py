import os
import subprocess
import time

def run_experiment(name, args):
    print(f"--- Running Experiment: {name} ---")
    print(f"Args: {args}")
    
    cmd = f".venv/bin/python main.py {args} --epochs 10 --dry_run" 
    # Note: epochs=10 and dry_run for testing the script. Real run should remove dry_run.
    # But for "Testing the Test Code", we keep it fast.
    # User can modify this script for full runs.
    
    # Let's run a short real run (not dry run) to generate logs?
    # dry_run exits after 1 step. That's good for verification.
    # But if we want actual metrics, we need more.
    # Let's stick to generating the script that *can* run it.
    
    # We will use a unique checkpoint path for each
    cmd += f" --checkpoint_path checkpoints/{name}.pt"
    
    # Redirect output
    log_file = f"experiments/{name}.log"
    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        process.wait()
        
    print(f"Experiment {name} finished. Log: {log_file}")

def main():
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # 1. Baseline
    run_experiment("baseline", "")
    
    # 2. Ablation: No Twin World
    run_experiment("ablation_no_twin_world", "--ablation_no_twin_world")
    
    # 3. Ablation: Dense MoE
    run_experiment("ablation_dense_moe", "--ablation_dense_moe")
    
    # 4. Ablation: No Interleaved (Additive)
    run_experiment("ablation_no_interleaved", "--ablation_no_interleaved")
    
    # 5. Ablation: Physics Only (No DAG Head)
    run_experiment("ablation_physics_only", "--ablation_no_dag")
    
    # 6. Ablation: Structure Only (No Physics Head)
    run_experiment("ablation_structure_only", "--ablation_no_physics")

if __name__ == "__main__":
    main()
