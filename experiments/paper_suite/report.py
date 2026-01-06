import pandas as pd
import os
import argparse

def generate_report(results_dir):
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    if not os.path.exists(csv_path):
        print(f"No results found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # 1. Summary by Model and N
    summary = df.groupby(["Model", "N"]).mean(numeric_only=True).reset_index()
    
    report_path = os.path.join(results_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("# Benchmark Report\n\n")
        
        f.write("## Overall Summary\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Detailed Results\n\n")
        f.write("### Structure Metrics (SHD, F1, SID, AUROC)\n\n")
        struct_cols = ["Model", "N", "SHD", "F1", "SID", "AUROC"]
        f.write(summary[struct_cols].to_markdown(index=False))
        f.write("\n\n")
        
        f.write("### Delta Metrics (MAE, MSE, RMSE, R2)\n\n")
        delta_cols = ["Model", "N", "MAE", "MSE", "RMSE", "R2"]
        f.write(summary[delta_cols].to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Runtime\n\n")
        time_cols = ["Model", "N", "Time_Struct", "Time_Delta"]
        f.write(summary[time_cols].to_markdown(index=False))
        f.write("\n\n")
    
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="experiments/paper_suite/results")
    args = parser.parse_args()
    generate_report(args.results_dir)
