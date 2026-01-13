import time
import os
import argparse
import sys
import numpy as np
import pandas as pd
import torch
import networkx as nx
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.models.CausalTransformer import CausalTransformer
from src.data.SCMGenerator import SCMGenerator
from src.training.metrics import compute_shd, compute_f1, compute_sid, compute_mae


def r2_score(pred: torch.Tensor, true: torch.Tensor) -> float:
    true_mean = true.mean()
    ss_res = torch.sum((pred - true) ** 2)
    ss_tot = torch.sum((true - true_mean) ** 2)
    if ss_tot.item() == 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def rmse(pred: torch.Tensor, true: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - true) ** 2)))


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = CausalTransformer(
        num_nodes=args.max_vars + 5,
        d_model=512,
        num_layers=args.num_layers,
        grad_checkpoint=getattr(args, "grad_checkpoint", False),
        ablation_dense=getattr(args, "ablation_dense_moe", False),
        ablation_no_interleaved=getattr(args, "ablation_no_interleaved", False),
        ablation_no_dag=getattr(args, "ablation_no_dag", False),
        ablation_no_physics=getattr(args, "ablation_no_physics", False),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, args


def evaluate_graph(model, generator: SCMGenerator, num_nodes: int, batch_size: int, device: torch.device, logit_threshold: float):
    pipe = generator.generate_pipeline(
        num_nodes=num_nodes,
        edge_prob=generator.edge_prob,
        num_samples_base=batch_size,
        num_samples_per_intervention=batch_size,
        intervention_prob=generator.intervention_prob,
        as_torch=True,
        use_twin_world=True,
    )

    base_tensor = pipe["base_tensor"][:batch_size].to(device)
    true_adj_np = nx.to_numpy_array(pipe["dag"], dtype=float)
    true_adj = torch.tensor(true_adj_np, dtype=torch.float32, device=device)

    graph_results = []

    # Iterate over each intervention setting
    for df_int, mask_arr in zip(pipe["all_dfs"][1:], pipe["all_masks"][1:]):
        int_samples = torch.tensor(df_int.values, dtype=torch.float32, device=device)[:batch_size]
        mask_vec = torch.tensor(mask_arr[0], dtype=torch.float32, device=device)  # shape (N,)
        mask = mask_vec.unsqueeze(0).repeat(batch_size, 1)
        int_idx = torch.argmax(mask_vec).repeat(batch_size)

        target_row = base_tensor  # twin-world target

        start = time.time()
        with torch.no_grad():
            deltas_pred, logits, _, _, _ = model(base_tensor, int_samples, target_row, mask)
        runtime = time.time() - start

        true_delta = int_samples - base_tensor
        logits_mean = logits.mean(dim=0)  # average over batch for structure metrics

        # Structure metrics
        shd = compute_shd(logits_mean.unsqueeze(0), true_adj.unsqueeze(0), threshold=logit_threshold)
        sid = compute_sid(logits_mean.unsqueeze(0), true_adj.unsqueeze(0), threshold=logit_threshold)
        f1 = compute_f1(logits_mean.unsqueeze(0), true_adj.unsqueeze(0), threshold=logit_threshold)

        try:
            probs = torch.sigmoid(logits_mean).detach().cpu().numpy().flatten()
            labels = true_adj.detach().cpu().numpy().flatten()
            auroc = float(roc_auc_score(labels, probs))
        except Exception:
            auroc = float("nan")

        # Delta metrics
        mae = compute_mae(deltas_pred, true_delta)
        mse = float(torch.mean((deltas_pred - true_delta) ** 2))
        delta_rmse = rmse(deltas_pred, true_delta)
        r2 = r2_score(deltas_pred, true_delta)

        graph_results.append({
            "N": num_nodes,
            "SHD": shd,
            "SID": sid,
            "F1": f1,
            "AUROC": auroc,
            "MAE": mae,
            "MSE": mse,
            "RMSE": delta_rmse,
            "R2": r2,
            "Time_s": runtime,
        })

    return graph_results


def run_full_eval(checkpoint_path: str, output_dir: str, device_name: str = "cpu", batch_size: int = 32, num_graphs: int = 5, sizes=None, logit_threshold: float = 1.33):
    if sizes is None:
        sizes = [10, 20, 30, 40, 50]

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_name)

    model, args = load_model(checkpoint_path, device)

    all_rows = []

    for n in sizes:
        print(f"[Eval] N={n} starting ...", flush=True)
        # Fresh generator per size with deterministic seed
        generator = SCMGenerator(num_nodes=n, edge_prob=0.2, seed=12345 + n, intervention_prob=0.2)
        for g_idx in range(num_graphs):
            # Advance seed for each graph
            generator.seed = 12345 + n * 100 + g_idx
            print(f"  - Graph {g_idx+1}/{num_graphs} (seed={generator.seed})", flush=True)
            graph_rows = evaluate_graph(model, generator, n, batch_size, device, logit_threshold)
            for row in graph_rows:
                row["Graph"] = g_idx
                all_rows.append(row)
            print(f"    completed: {len(graph_rows)} interventions", flush=True)

    df = pd.DataFrame(all_rows)
    raw_path = os.path.join(output_dir, "full_eval_static.csv")
    df.to_csv(raw_path, index=False)

    summary = df.groupby("N").mean(numeric_only=True).reset_index()
    summary_path = os.path.join(output_dir, "full_eval_static_summary.csv")
    summary.to_csv(summary_path, index=False)

    md_path = os.path.join(output_dir, "full_eval_static.md")
    with open(md_path, "w") as f:
        f.write("# Full Static Evaluation\n\n")
        f.write("## Per-Intervention Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Summary by N\n\n")
        f.write(summary.to_markdown(index=False))

    return raw_path, summary_path, md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch_253.pt")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_graphs", type=int, default=5)
    parser.add_argument("--sizes", type=str, default="10,20,30,40,50")
    parser.add_argument("--logit_threshold", type=float, default=1.33)
    args = parser.parse_args()

    sizes = [int(x) for x in args.sizes.split(',') if x]
    run_full_eval(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        num_graphs=args.num_graphs,
        sizes=sizes,
        logit_threshold=args.logit_threshold,
    )


if __name__ == "__main__":
    main()
