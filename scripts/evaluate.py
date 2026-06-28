#!/usr/bin/env python
"""Evaluation script for trained Adaptive Causal Model."""

import argparse
import json
import torch
import numpy as np
from pathlib import Path

from crt.integration.adaptive_model import AdaptiveCausalModel
from crt.data.synthetic import SyntheticCausalDataset
from crt.utils.metrics import (
    structural_hamming_distance,
    edge_precision_recall_f1,
    tpr_fdr,
)
from crt.utils.visualization import (
    plot_graph_comparison,
    plot_adjacency_heatmaps,
    plot_surprise_evolution,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Adaptive Causal Model")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num-eval-graphs", type=int, default=100)
    p.add_argument("--output-dir", type=str, default="eval_output")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--num-vis-examples", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


@torch.no_grad()
def evaluate_structure(model, dataset, device, threshold=0.5):
    model.eval()
    all_shd = []
    all_precision = []
    all_recall = []
    all_f1 = []
    all_tpr = []
    all_fdr = []

    for i in range(len(dataset)):
        item = dataset[i]
        obs = item["observations"].mean(0, keepdim=True).to(device)
        int_data = item["interventions"].unsqueeze(0).to(device)
        int_values = item["intervention_values"].unsqueeze(0).to(device)
        true_adj = item["adjacency"]

        output = model(obs, int_data, int_values)
        pred_adj = output["final_adj"].mean(0).cpu()

        all_shd.append(structural_hamming_distance(pred_adj, true_adj, threshold))
        prf = edge_precision_recall_f1(pred_adj, true_adj, threshold)
        all_precision.append(prf["precision"])
        all_recall.append(prf["recall"])
        all_f1.append(prf["f1"])
        td = tpr_fdr(pred_adj, true_adj, threshold)
        all_tpr.append(td["tpr"])
        all_fdr.append(td["fdr"])

    return {
        "shd_mean": float(np.mean(all_shd)),
        "shd_std": float(np.std(all_shd)),
        "precision_mean": float(np.mean(all_precision)),
        "precision_std": float(np.std(all_precision)),
        "recall_mean": float(np.mean(all_recall)),
        "recall_std": float(np.std(all_recall)),
        "f1_mean": float(np.mean(all_f1)),
        "f1_std": float(np.std(all_f1)),
        "tpr_mean": float(np.mean(all_tpr)),
        "tpr_std": float(np.std(all_tpr)),
        "fdr_mean": float(np.mean(all_fdr)),
        "fdr_std": float(np.std(all_fdr)),
        "num_graphs": len(dataset),
    }


@torch.no_grad()
def generate_visualizations(model, dataset, device, output_dir, threshold=0.5,
                            num_examples=5):
    model.eval()
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(num_examples, len(dataset))):
        item = dataset[i]
        obs = item["observations"].mean(0, keepdim=True).to(device)
        int_data = item["interventions"].unsqueeze(0).to(device)
        int_values = item["intervention_values"].unsqueeze(0).to(device)
        true_adj = item["adjacency"]

        output = model(obs, int_data, int_values)
        pred_adj = output["final_adj"].mean(0).cpu()

        plot_graph_comparison(
            true_adj, pred_adj, threshold=threshold,
            save_path=str(figures_dir / f"graph_comparison_{i}.png"))

        adjacencies = [a.cpu() for a in output["adjacencies"]]
        plot_adjacency_heatmaps(
            adjacencies, true_adj=true_adj,
            save_path=str(figures_dir / f"adjacency_evolution_{i}.png"))

        if output["surprises"]:
            surprises = [s.cpu() for s in output["surprises"]]
            plot_surprise_evolution(
                surprises,
                save_path=str(figures_dir / f"surprise_evolution_{i}.png"))

        plt_close_all()

    print(f"Saved {min(num_examples, len(dataset))} visualization sets to {figures_dir}")


def plt_close_all():
    import matplotlib.pyplot as plt
    plt.close('all')


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = checkpoint["args"]

    model = AdaptiveCausalModel(
        num_nodes=train_args["num_nodes"],
        num_iterations=train_args["num_iterations"],
        hidden_dim=train_args["hidden_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    print("Generating evaluation dataset...")
    dataset = SyntheticCausalDataset(
        num_graphs=args.num_eval_graphs,
        num_nodes=train_args["num_nodes"],
        edge_prob=train_args["edge_prob"],
        equation_type=train_args["equation_type"],
        noise_std=train_args["noise_std"],
        seed=args.seed)

    print("Evaluating structure learning...")
    metrics = evaluate_structure(model, dataset, device, args.threshold)

    print("\n" + "=" * 50)
    print("Structure Learning Metrics")
    print("=" * 50)
    print(f"  SHD:       {metrics['shd_mean']:.2f} ± {metrics['shd_std']:.2f}")
    print(f"  Precision: {metrics['precision_mean']:.3f} ± {metrics['precision_std']:.3f}")
    print(f"  Recall:    {metrics['recall_mean']:.3f} ± {metrics['recall_std']:.3f}")
    print(f"  F1:        {metrics['f1_mean']:.3f} ± {metrics['f1_std']:.3f}")
    print(f"  TPR:       {metrics['tpr_mean']:.3f} ± {metrics['tpr_std']:.3f}")
    print(f"  FDR:       {metrics['fdr_mean']:.3f} ± {metrics['fdr_std']:.3f}")
    print("=" * 50)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {output_dir / 'metrics.json'}")

    print("Generating visualizations...")
    generate_visualizations(model, dataset, device, args.output_dir,
                            args.threshold, args.num_vis_examples)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
