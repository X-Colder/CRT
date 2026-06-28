"""
Adaptive Causal Model — the unified closed-loop architecture.

Iterative cycle: predict → detect surprise → revise graph → verify.
Unifies all three research paths into a single "按图索骥 → 因骥改图" loop.

Supports both flat graphs (AdaptiveCausalModel) and hierarchical
multi-level graphs (HierarchicalAdaptiveModel).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from crt.graph.causal_graph import DifferentiableCausalGraph
from crt.graph.hierarchical_graph import HierarchicalCausalGraph


class SurpriseDetector(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int = 64):
        super().__init__()
        self.surprise_net = nn.Sequential(
            nn.Linear(num_nodes * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(num_nodes, 1),
            nn.Sigmoid(),
        )

    def forward(self, predicted: torch.Tensor,
                observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = observed - predicted
        combined = torch.cat([residual, observed], dim=-1)
        surprise = self.surprise_net(combined)
        gate = self.gate_net(surprise)
        return surprise, gate


class GraphReviser(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int = 64):
        super().__init__()
        self.num_nodes = num_nodes
        self.pairwise_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, adjacency: torch.Tensor, surprise: torch.Tensor,
                node_values: torch.Tensor,
                gate: torch.Tensor) -> torch.Tensor:
        B, N = surprise.shape

        s_i = surprise.unsqueeze(2).expand(-1, -1, N)
        s_j = surprise.unsqueeze(1).expand(-1, N, -1)
        v_i = node_values.unsqueeze(2).expand(-1, -1, N)
        v_j = node_values.unsqueeze(1).expand(-1, N, -1)

        features = torch.stack([adjacency, s_i, s_j, v_i, v_j], dim=-1)
        delta = self.pairwise_mlp(features).squeeze(-1)
        delta = torch.tanh(delta) * gate.unsqueeze(-1)

        adj_logits = torch.logit(adjacency.clamp(1e-6, 1 - 1e-6))
        new_adj = torch.sigmoid(adj_logits + delta)

        mask = 1 - torch.eye(N, device=new_adj.device)
        return new_adj * mask.unsqueeze(0)


class AdaptiveCausalModel(nn.Module):
    def __init__(self, num_nodes: int, num_iterations: int = 3,
                 hidden_dim: int = 64):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_iterations = num_iterations

        self.causal_graph = DifferentiableCausalGraph(num_nodes)
        self.surprise_detector = SurpriseDetector(num_nodes, hidden_dim)
        self.graph_reviser = GraphReviser(num_nodes, hidden_dim)

    def _batched_forward(self, x: torch.Tensor,
                         adj: torch.Tensor) -> torch.Tensor:
        B, N = x.shape
        in_degrees = adj.mean(0).sum(dim=0)
        order = torch.argsort(in_degrees).tolist()

        result = torch.zeros_like(x)
        for idx in order:
            parent_values = x * adj[:, :, idx]
            result[:, idx] = self.causal_graph.structural_eqs[idx](parent_values).squeeze(-1)
        return result

    def forward(self, observations: torch.Tensor,
                interventions: Optional[torch.Tensor] = None,
                intervention_values: Optional[torch.Tensor] = None
                ) -> dict[str, torch.Tensor]:
        B = observations.shape[0]
        adj = self.causal_graph.adjacency.unsqueeze(0).expand(B, -1, -1)

        predictions = []
        adjacencies = [adj]
        surprises = []
        gates = []

        for _ in range(self.num_iterations):
            pred = self._batched_forward(observations, adj)
            predictions.append(pred)

            surprise, gate = self.surprise_detector(pred, observations)
            surprises.append(surprise)
            gates.append(gate)

            adj = self.graph_reviser(adj, surprise, observations, gate)
            adjacencies.append(adj)

        final_pred = self._batched_forward(observations, adj)
        predictions.append(final_pred)

        mean_adj = adj.mean(dim=0)
        d = self.num_nodes
        M = torch.eye(d, device=mean_adj.device) + mean_adj / d
        E = torch.matrix_power(M, d)
        dag_penalty = torch.trace(E) - d

        result = {
            "predictions": predictions,
            "adjacencies": adjacencies,
            "surprises": surprises,
            "gates": gates,
            "dag_penalty": dag_penalty,
            "final_adj": adj,
        }

        if interventions is not None and intervention_values is not None:
            result["intervention_loss"] = self._intervention_loss(
                adj, observations, interventions, intervention_values)

        return result

    def _intervention_loss(self, adj: torch.Tensor,
                           observations: torch.Tensor,
                           interventions: torch.Tensor,
                           intervention_values: torch.Tensor) -> torch.Tensor:
        B, N = observations.shape
        total_loss = torch.tensor(0.0, device=observations.device)

        for k in range(N):
            adj_int = adj.clone()
            adj_int[:, :, k] = 0

            int_obs = observations.clone()
            int_obs[:, k] = intervention_values[:, k]

            pred_int = self._batched_forward(int_obs, adj_int)
            target_int = interventions[:, k].mean(dim=1)

            mask = torch.ones(N, device=observations.device)
            mask[k] = 0
            diff = (pred_int - target_int) * mask.unsqueeze(0)
            total_loss = total_loss + diff.pow(2).mean()

        return total_loss / N

    def loss(self, output: dict,
             true_adj: torch.Tensor,
             observations: torch.Tensor,
             interventions: Optional[torch.Tensor] = None,
             intervention_values: Optional[torch.Tensor] = None,
             weights: Optional[dict[str, float]] = None) -> dict[str, torch.Tensor]:
        w = {
            "prediction": 1.0, "adjacency": 1.0, "intervention": 0.5,
            "dag": 0.1, "sparsity": 0.05, "convergence": 0.1,
        }
        if weights:
            w.update(weights)

        device = observations.device
        final_pred = output["predictions"][-1]
        pred_loss = F.mse_loss(final_pred, observations)

        final_adj = output["final_adj"]
        true_adj_expanded = true_adj.unsqueeze(0).expand_as(final_adj)
        adj_loss = F.binary_cross_entropy(
            final_adj.clamp(1e-6, 1 - 1e-6), true_adj_expanded)

        int_loss = output.get("intervention_loss", torch.tensor(0.0, device=device))
        dag_loss = output["dag_penalty"]

        sparsity_loss = final_adj.mean()

        adjs = output["adjacencies"]
        conv_loss = torch.tensor(0.0, device=device)
        if len(adjs) >= 3:
            deltas = []
            for t in range(1, len(adjs)):
                deltas.append((adjs[t] - adjs[t - 1]).pow(2).sum(dim=(-2, -1)).mean())
            for t in range(1, len(deltas)):
                conv_loss = conv_loss + F.relu(deltas[t] - deltas[t - 1])
            conv_loss = conv_loss / max(len(deltas) - 1, 1)

        total = (w["prediction"] * pred_loss
                 + w["adjacency"] * adj_loss
                 + w["intervention"] * int_loss
                 + w["dag"] * dag_loss
                 + w["sparsity"] * sparsity_loss
                 + w["convergence"] * conv_loss)

        return {
            "total": total,
            "prediction": pred_loss,
            "adjacency": adj_loss,
            "intervention": int_loss,
            "dag": dag_loss,
            "sparsity": sparsity_loss,
            "convergence": conv_loss,
        }


class HierarchicalAdaptiveModel(nn.Module):
    """
    Hierarchical version of the adaptive causal model.

    Each abstraction level has its own surprise detector and graph reviser.
    Cross-level bridges propagate information bottom-up (aggregate evidence)
    and top-down (propagate causal context).

    The revision loop runs across ALL levels simultaneously:
    1. Intra-level prediction at each level
    2. Bottom-up aggregation (micro surprise informs macro revision)
    3. Top-down grounding (macro structure constrains micro revision)
    4. Per-level graph revision
    """

    def __init__(self, level_sizes: list[int], num_iterations: int = 3,
                 hidden_dim: int = 64):
        super().__init__()
        self.level_sizes = level_sizes
        self.num_levels = len(level_sizes)
        self.num_iterations = num_iterations
        self.total_nodes = sum(level_sizes)

        self.hierarchical_graph = HierarchicalCausalGraph(level_sizes, hidden_dim)

        self.level_surprise = nn.ModuleList([
            SurpriseDetector(n, hidden_dim) for n in level_sizes
        ])
        self.level_reviser = nn.ModuleList([
            GraphReviser(n, hidden_dim) for n in level_sizes
        ])

    def _split_by_level(self, x: torch.Tensor) -> list[torch.Tensor]:
        parts = []
        offset = 0
        for size in self.level_sizes:
            parts.append(x[:, offset:offset + size])
            offset += size
        return parts

    def _batched_forward_level(self, x: torch.Tensor, adj: torch.Tensor,
                               level_idx: int) -> torch.Tensor:
        B, N = x.shape
        level = self.hierarchical_graph.levels[level_idx]
        in_degrees = adj.mean(0).sum(dim=0)
        order = torch.argsort(in_degrees).tolist()

        result = torch.zeros_like(x)
        for idx in order:
            parent_values = x * adj[:, :, idx]
            result[:, idx] = level.structural_eqs[idx](parent_values).squeeze(-1)
        return result

    def forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        B = observations.shape[0]
        level_obs = self._split_by_level(observations)

        level_adjs = []
        for i, level in enumerate(self.hierarchical_graph.levels):
            level_adjs.append(
                level.adjacency.unsqueeze(0).expand(B, -1, -1))

        all_predictions = []
        all_adjacencies = [self._collect_adjs(level_adjs)]
        all_surprises = []
        all_gates = []

        for _ in range(self.num_iterations):
            level_preds = []
            level_surprise_vals = []
            level_gate_vals = []

            for i in range(self.num_levels):
                pred = self._batched_forward_level(level_obs[i], level_adjs[i], i)
                level_preds.append(pred)
                surprise, gate = self.level_surprise[i](pred, level_obs[i])
                level_surprise_vals.append(surprise)
                level_gate_vals.append(gate)

            for i in range(self.num_levels - 1):
                bridge = self.hierarchical_graph.bridges[i]
                level_surprise_vals[i + 1] = bridge.bottom_up(
                    level_surprise_vals[i], level_surprise_vals[i + 1])

            for i in range(self.num_levels - 2, -1, -1):
                bridge = self.hierarchical_graph.bridges[i]
                level_surprise_vals[i] = bridge.top_down(
                    level_surprise_vals[i + 1], level_surprise_vals[i])

            for i in range(self.num_levels):
                level_adjs[i] = self.level_reviser[i](
                    level_adjs[i], level_surprise_vals[i],
                    level_obs[i], level_gate_vals[i])

            all_predictions.append(torch.cat(level_preds, dim=-1))
            all_adjacencies.append(self._collect_adjs(level_adjs))
            all_surprises.append(torch.cat(level_surprise_vals, dim=-1))
            all_gates.append(torch.cat(level_gate_vals, dim=-1))

        final_preds = []
        for i in range(self.num_levels):
            final_preds.append(
                self._batched_forward_level(level_obs[i], level_adjs[i], i))
        all_predictions.append(torch.cat(final_preds, dim=-1))

        return {
            "predictions": all_predictions,
            "adjacencies": all_adjacencies,
            "surprises": all_surprises,
            "gates": all_gates,
            "dag_penalty": self.hierarchical_graph.dag_penalty(),
            "final_adj": self._build_full_adj(level_adjs),
            "level_adjs": level_adjs,
        }

    def _collect_adjs(self, level_adjs: list[torch.Tensor]) -> torch.Tensor:
        return self._build_full_adj(level_adjs)

    def _build_full_adj(self, level_adjs: list[torch.Tensor]) -> torch.Tensor:
        B = level_adjs[0].shape[0]
        N = self.total_nodes
        device = level_adjs[0].device
        full = torch.zeros(B, N, N, device=device)
        offset = 0
        for adj in level_adjs:
            n = adj.shape[-1]
            full[:, offset:offset + n, offset:offset + n] = adj
            offset += n
        return full

    def loss(self, output: dict,
             true_adj: torch.Tensor,
             observations: torch.Tensor,
             weights: Optional[dict[str, float]] = None) -> dict[str, torch.Tensor]:
        w = {
            "prediction": 1.0, "adjacency": 1.0,
            "dag": 0.1, "sparsity": 0.05, "convergence": 0.1,
        }
        if weights:
            w.update(weights)

        device = observations.device
        final_pred = output["predictions"][-1]
        pred_loss = F.mse_loss(final_pred, observations)

        final_adj = output["final_adj"]
        true_adj_expanded = true_adj.unsqueeze(0).expand_as(final_adj)
        adj_loss = F.binary_cross_entropy(
            final_adj.clamp(1e-6, 1 - 1e-6), true_adj_expanded)

        dag_loss = output["dag_penalty"]
        sparsity_loss = final_adj.mean()

        adjs = output["adjacencies"]
        conv_loss = torch.tensor(0.0, device=device)
        if len(adjs) >= 3:
            deltas = []
            for t in range(1, len(adjs)):
                deltas.append((adjs[t] - adjs[t - 1]).pow(2).sum(dim=(-2, -1)).mean())
            for t in range(1, len(deltas)):
                conv_loss = conv_loss + F.relu(deltas[t] - deltas[t - 1])
            conv_loss = conv_loss / max(len(deltas) - 1, 1)

        total = (w["prediction"] * pred_loss
                 + w["adjacency"] * adj_loss
                 + w["dag"] * dag_loss
                 + w["sparsity"] * sparsity_loss
                 + w["convergence"] * conv_loss)

        return {
            "total": total,
            "prediction": pred_loss,
            "adjacency": adj_loss,
            "dag": dag_loss,
            "sparsity": sparsity_loss,
            "convergence": conv_loss,
        }
