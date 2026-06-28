"""
Evaluation metrics for causal structure learning.
"""

import torch


def structural_hamming_distance(pred_adj: torch.Tensor,
                                true_adj: torch.Tensor,
                                threshold: float = 0.5) -> int:
    pred_bin = (pred_adj > threshold).int()
    true_bin = true_adj.int()
    diff = pred_bin - true_bin
    extra = (diff == 1).sum()
    missing = (diff == -1).sum()
    reversed_edges = ((pred_bin == 1) & (pred_bin.T == 0) &
                      (true_bin == 0) & (true_bin.T == 1)).sum()
    return (extra + missing + reversed_edges).item()


def edge_precision_recall_f1(pred_adj: torch.Tensor,
                             true_adj: torch.Tensor,
                             threshold: float = 0.5) -> dict[str, float]:
    pred_bin = (pred_adj > threshold).float()
    true_bin = true_adj.float()
    eps = 1e-8
    tp = (pred_bin * true_bin).sum()
    fp = (pred_bin * (1 - true_bin)).sum()
    fn = ((1 - pred_bin) * true_bin).sum()
    precision = (tp / (tp + fp + eps)).item()
    recall = (tp / (tp + fn + eps)).item()
    f1 = (2 * precision * recall / (precision + recall + eps))
    return {"precision": precision, "recall": recall, "f1": f1}


def tpr_fdr(pred_adj: torch.Tensor,
            true_adj: torch.Tensor,
            threshold: float = 0.5) -> dict[str, float]:
    metrics = edge_precision_recall_f1(pred_adj, true_adj, threshold)
    return {
        "tpr": metrics["recall"],
        "fdr": 1.0 - metrics["precision"],
    }
