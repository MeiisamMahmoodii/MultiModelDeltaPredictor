import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.data.SCMGenerator import SCMGenerator
from src.training.metrics import compute_shd, compute_f1, compute_sid
from src.models.baselines.wrappers import PCWrapper, NotearsWrapper


def run_baselines(output_dir: str, batch_size: int = 200, num_graphs: int = 3, sizes=None, device: str = "cpu", logit_threshold: float = 0.0):
    if sizes is None:
        sizes = [10, 20, 30, 40, 50]

    os.makedirs(output_dir, exist_ok=True)

    rows = []

    for n in sizes:
        print(f"[Baseline Eval] N={n} starting ...", flush=True)
        generator = SCMGenerator(num_nodes=n, edge_prob=0.2, seed=12345 + n, intervention_prob=0.2)
        for g_idx in range(num_graphs):
            generator.seed = 12345 + n * 100 + g_idx
            print(f"  - Graph {g_idx+1}/{num_graphs} (seed={generator.seed})", flush=True)

            pipe = generator.generate_pipeline(
                num_nodes=n,
                edge_prob=generator.edge_prob,
                num_samples_base=batch_size,
                num_samples_per_intervention=batch_size,
                intervention_prob=generator.intervention_prob,
                as_torch=True,
                use_twin_world=True,
            )

            base_tensor = pipe["base_tensor"][:batch_size]
            X = base_tensor.numpy()
            true_adj_np = generator.edge_parameters(pipe["dag"])
            import networkx as nx
            true_adj_np = nx.to_numpy_array(true_adj_np, dtype=float)
            true_adj = torch.tensor(true_adj_np, dtype=torch.float32)
            true_adj = (true_adj != 0).float()

            # PC (skip for large N to avoid long runs; limit conditioning set)
            if n <= 30:
                try:
                    start = time.time()
                    pc_model = PCWrapper(alpha=0.05, max_k=2)
                    pc_model.fit(X)
                    pc_adj = torch.tensor(pc_model.predict_adj(), dtype=torch.float32)
                    pc_time = time.time() - start

                    pc_shd = compute_shd(pc_adj.unsqueeze(0), true_adj.unsqueeze(0), threshold=logit_threshold)
                    pc_sid = compute_sid(pc_adj.unsqueeze(0), true_adj.unsqueeze(0), threshold=logit_threshold)
                    pc_f1 = compute_f1(pc_adj.unsqueeze(0), true_adj.unsqueeze(0), threshold=logit_threshold)

                    rows.append({
                        "Model": "PC",
                        "N": n,
                        "Graph": g_idx,
                        "SHD": pc_shd,
                        "SID": pc_sid,
                        "F1": pc_f1,
                        "MAE": np.nan,
                        "MSE": np.nan,
                        "RMSE": np.nan,
                        "R2": np.nan,
                        "Time_s": pc_time,
                    })
                    print(f"    PC done: SHD={pc_shd:.2f}, F1={pc_f1:.3f}, time={pc_time:.2f}s", flush=True)
                except Exception as e:
                    rows.append({"Model": "PC", "N": n, "Graph": g_idx, "Error": str(e)})
                    print(f"    PC failed: {e}", flush=True)
            else:
                rows.append({"Model": "PC", "N": n, "Graph": g_idx, "Error": "skipped_n_gt_30"})
                print("    PC skipped (N>30)", flush=True)

            # NOTEARS (linear)
            try:
                start = time.time()
                nt_model = NotearsWrapper(use_gpu=(device != "cpu"))
                nt_model.fit(X)
                nt_adj = torch.tensor(nt_model.predict_adj(), dtype=torch.float32)
                nt_time = time.time() - start

                nt_shd = compute_shd(nt_adj.unsqueeze(0), true_adj.unsqueeze(0), threshold=logit_threshold)
                nt_sid = compute_sid(nt_adj.unsqueeze(0), true_adj.unsqueeze(0), threshold=logit_threshold)
                nt_f1 = compute_f1(nt_adj.unsqueeze(0), true_adj.unsqueeze(0), threshold=logit_threshold)

                rows.append({
                    "Model": "NOTEARS",
                    "N": n,
                    "Graph": g_idx,
                    "SHD": nt_shd,
                    "SID": nt_sid,
                    "F1": nt_f1,
                    "MAE": np.nan,
                    "MSE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "Time_s": nt_time,
                })
                print(f"    NOTEARS done: SHD={nt_shd:.2f}, F1={nt_f1:.3f}, time={nt_time:.2f}s", flush=True)
            except Exception as e:
                rows.append({"Model": "NOTEARS", "N": n, "Graph": g_idx, "Error": str(e)})
                print(f"    NOTEARS failed: {e}", flush=True)

    df = pd.DataFrame(rows)
    raw_path = os.path.join(output_dir, "full_eval_baselines.csv")
    df.to_csv(raw_path, index=False)

    summary = df.groupby(["Model", "N"]).mean(numeric_only=True).reset_index()
    summary_path = os.path.join(output_dir, "full_eval_baselines_summary.csv")
    summary.to_csv(summary_path, index=False)

    md_path = os.path.join(output_dir, "full_eval_baselines.md")
    with open(md_path, "w") as f:
        f.write("# Baseline Evaluation\n\n")
        f.write("## Per-Graph Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Summary\n\n")
        f.write(summary.to_markdown(index=False))

    print(f"Saved: {raw_path} and {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_graphs", type=int, default=3)
    parser.add_argument("--sizes", type=str, default="10,20,30,40,50")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    sizes = [int(x) for x in args.sizes.split(',') if x]
    run_baselines(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_graphs=args.num_graphs,
        sizes=sizes,
        device=args.device,
    )


if __name__ == "__main__":
    main()
