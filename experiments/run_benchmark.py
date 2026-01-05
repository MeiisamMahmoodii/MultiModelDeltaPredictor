import os
import subprocess

def run_benchmark(name, args):
    print(f"--- Running Benchmark: {name} ---")
    cmd = f".venv/bin/python main.py {args} --dry_run"
    log_file = f"experiments/{name}.log"
    with open(log_file, "w") as f:
        subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    print(f"Benchmark {name} finished.")

def main():
    os.makedirs("experiments", exist_ok=True)
    
    # Comparison tiers
    # Tier 1: Existing Methods (Not run here, require external code)
    # This script runs OUR model in different configurations to mimic others or scaling.
    
    # Scaling Test
    scales = [20, 50, 100]
    for s in scales:
        run_benchmark(f"scale_{s}", f"--min_vars {s} --max_vars {s} --batch_size 8")

if __name__ == "__main__":
    main()
