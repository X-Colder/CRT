"""
Visualization tools for causal structure learning.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import Optional


def plot_graph_comparison(true_adj: torch.Tensor,
                          pred_adj: torch.Tensor,
                          node_names: Optional[list[str]] = None,
                          threshold: float = 0.5,
                          save_path: Optional[str] = None) -> plt.Figure:
    N = true_adj.shape[0]
    if node_names is None:
        node_names = [f"X{i}" for i in range(N)]

    pred_bin = (pred_adj > threshold).int()
    true_bin = true_adj.int()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, adj, title in [(axes[0], true_bin, "Ground Truth"),
                           (axes[1], pred_bin, "Discovered")]:
        g = nx.DiGraph()
        g.add_nodes_from(range(N))
        edge_colors = []
        edge_widths = []
        for i in range(N):
            for j in range(N):
                if adj[i, j] > 0:
                    g.add_edge(i, j)
                    if true_bin[i, j] > 0 and pred_bin[i, j] > 0:
                        edge_colors.append('#2ecc71')
                        edge_widths.append(2.0)
                    elif pred_bin[i, j] > 0 and true_bin[i, j] == 0:
                        edge_colors.append('#e74c3c')
                        edge_widths.append(1.5)
                    else:
                        edge_colors.append('#7f8c8d')
                        edge_widths.append(1.0)

        pos = nx.spring_layout(g, seed=42)
        nx.draw(g, pos, ax=ax, with_labels=True,
                labels={i: node_names[i] for i in range(N)},
                node_color='#3498db', node_size=500, font_color='white',
                font_weight='bold', edge_color=edge_colors,
                width=edge_widths if edge_widths else 1.0,
                arrowsize=15, arrowstyle='->')
        ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_adjacency_heatmaps(adjacencies: list[torch.Tensor],
                            true_adj: Optional[torch.Tensor] = None,
                            node_names: Optional[list[str]] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
    panels = []
    titles = []
    if true_adj is not None:
        panels.append(true_adj.detach().cpu().numpy())
        titles.append("Ground Truth")
    for i, adj in enumerate(adjacencies):
        a = adj.detach().cpu()
        if a.dim() == 3:
            a = a.mean(0)
        panels.append(a.numpy())
        titles.append(f"Iteration {i}")

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    N = panels[0].shape[0]
    if node_names is None:
        node_names = [f"X{i}" for i in range(N)]

    for ax, data, title in zip(axes, panels, titles):
        im = ax.imshow(data, cmap='RdBu_r', vmin=0, vmax=1, aspect='equal')
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(node_names, fontsize=7, rotation=45)
        ax.set_yticklabels(node_names, fontsize=7)
        ax.set_title(title, fontsize=11, fontweight='bold')

    fig.colorbar(im, ax=axes, shrink=0.8, label='Edge weight')
    fig.subplots_adjust(wspace=0.3)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_surprise_evolution(surprises: list[torch.Tensor],
                            node_names: Optional[list[str]] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
    data = []
    for s in surprises:
        s_cpu = s.detach().cpu()
        if s_cpu.dim() == 2:
            s_cpu = s_cpu.mean(0)
        data.append(s_cpu.numpy())
    data = np.array(data)
    N = data.shape[1]

    if node_names is None:
        node_names = [f"X{i}" for i in range(N)]

    fig, ax = plt.subplots(figsize=(10, 6))
    iterations = range(len(surprises))
    for i in range(N):
        ax.plot(iterations, data[:, i], '-o', markersize=4,
                alpha=0.6, label=node_names[i])
    ax.plot(iterations, data.mean(axis=1), 'k-s', markersize=6,
            linewidth=2.5, label='Mean', zorder=10)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Surprise', fontsize=12)
    ax.set_title('Surprise Evolution Across Iterations', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_training_metrics(metrics_history: dict[str, list[float]],
                          save_path: Optional[str] = None) -> plt.Figure:
    keys = [k for k in metrics_history if len(metrics_history[k]) > 0]
    n_plots = len(keys)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i, key in enumerate(keys):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        values = metrics_history[key]
        ax.plot(values, linewidth=1.5)
        ax.set_title(key.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)

    for i in range(n_plots, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
