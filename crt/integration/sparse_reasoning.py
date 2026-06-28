"""
Sparse Reasoning Engine — "think along the shortest causal path".

Human reasoning is efficient because it activates only a few relevant
variables, reasons along the shortest causal chain, and expands only
when the current explanation is insufficient.

This module implements five core components:

1. RelevanceScorer — "What nodes matter for this question?"
2. SufficiencyDetector — "Can these nodes explain what I see?"
3. ExpansionStrategy — "Which direction should I search next?"
4. ConvergenceDetector — "Should I stop thinking?"
   Three convergence criteria:
   (a) Structural: causal chain reached root/terminal nodes
   (b) Predictive: residual is small enough
   (c) Marginal: expanding further yields diminishing returns
5. SparseReasoningEngine — Orchestrates the full adaptive loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from crt.graph.causal_graph import DifferentiableCausalGraph


class RelevanceScorer(nn.Module):
    """
    Scores each node's causal relevance to the current query.

    Not semantic similarity — causal relevance: "Would intervening
    on this node change my conclusion?"
    """

    def __init__(self, num_nodes: int, query_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.node_proj = nn.Linear(1, hidden_dim)
        self.centrality_proj = nn.Linear(2, hidden_dim)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query: torch.Tensor, node_values: torch.Tensor,
                adjacency: torch.Tensor) -> torch.Tensor:
        B, N = node_values.shape
        q = self.query_proj(query).unsqueeze(1).expand(-1, N, -1)
        v = self.node_proj(node_values.unsqueeze(-1))

        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0).expand(B, -1, -1)
        in_degree = adjacency.sum(dim=1)
        out_degree = adjacency.sum(dim=2)
        centrality_features = torch.stack([in_degree, out_degree], dim=-1)
        c = self.centrality_proj(centrality_features)

        combined = torch.cat([q, v, c], dim=-1)
        return torch.sigmoid(self.score_head(combined).squeeze(-1))


class SufficiencyDetector(nn.Module):
    """
    Detects whether the currently active nodes can adequately
    explain the observations.
    """

    def __init__(self, num_nodes: int, hidden_dim: int = 64):
        super().__init__()
        self.residual_net = nn.Sequential(
            nn.Linear(num_nodes * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes),
        )
        self.sufficiency_gate = nn.Sequential(
            nn.Linear(num_nodes, 1),
            nn.Sigmoid(),
        )

    def forward(self, predicted: torch.Tensor, observed: torch.Tensor,
                active_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = (observed - predicted).abs()
        masked_residual = residual * active_mask
        combined = torch.cat([masked_residual, observed], dim=-1)
        insufficiency = self.residual_net(combined)
        sufficient = self.sufficiency_gate(insufficiency.abs())
        return sufficient, insufficiency


class ExpansionStrategy(nn.Module):
    """
    Decides which INACTIVE nodes to activate next, guided by
    insufficiency location, causal proximity, and direction.
    """

    def __init__(self, num_nodes: int, hidden_dim: int = 64):
        super().__init__()
        self.expansion_net = nn.Sequential(
            nn.Linear(num_nodes * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(num_nodes, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, insufficiency: torch.Tensor, active_mask: torch.Tensor,
                adjacency: torch.Tensor, node_values: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N = insufficiency.shape

        direction = self.direction_head(insufficiency)
        upstream_w = direction[:, 0:1]
        downstream_w = direction[:, 1:2]

        upstream_neighbors = torch.bmm(
            active_mask.unsqueeze(1), adjacency.transpose(1, 2)).squeeze(1)
        downstream_neighbors = torch.bmm(
            active_mask.unsqueeze(1), adjacency).squeeze(1)

        causal_proximity = (upstream_w * upstream_neighbors +
                            downstream_w * downstream_neighbors)

        combined = torch.cat([insufficiency, causal_proximity, node_values], dim=-1)
        expansion_scores = torch.sigmoid(self.expansion_net(combined))

        inactive = 1 - active_mask
        expansion = expansion_scores * inactive

        new_mask = (active_mask + expansion).clamp(0, 1)
        return new_mask, direction


class ConvergenceDetector(nn.Module):
    """
    Determines whether reasoning should stop. Three convergence criteria:

    1. Structural closure — have we reached root nodes (no parents) and
       terminal nodes (no children)? A causal chain that "hangs" in the
       middle without reaching either end is incomplete.

    2. Predictive adequacy — is the prediction residual small enough?
       The active subgraph can explain the observations well.

    3. Marginal diminishment — would expanding further actually help?
       If the last expansion barely changed the prediction, more
       expansion is unlikely to help either.

    The three signals are fused into a single soft halting probability
    h_t in [0, 1]. The reasoning loop uses Adaptive Computation Time
    (ACT, Graves 2016): accumulate h_t across steps, halt when the
    cumulative sum exceeds 1. This makes the number of reasoning steps
    a LEARNED, input-dependent quantity.
    """

    def __init__(self, num_nodes: int, hidden_dim: int = 64):
        super().__init__()
        self.num_nodes = num_nodes

        self.structural_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.predictive_net = nn.Sequential(
            nn.Linear(num_nodes + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.marginal_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.fusion = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, active_mask: torch.Tensor, adjacency: torch.Tensor,
                predicted: torch.Tensor, observed: torch.Tensor,
                prev_predicted: Optional[torch.Tensor],
                sufficiency: torch.Tensor
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            active_mask:    (B, N) current activation.
            adjacency:      (B, N, N) causal adjacency.
            predicted:      (B, N) current prediction.
            observed:       (B, N) observation.
            prev_predicted: (B, N) prediction from previous step (or None).
            sufficiency:    (B, 1) sufficiency score from detector.
        Returns:
            halt_prob: (B, 1) probability of stopping at this step.
            details:   dict with per-criterion scores.
        """
        B, N = active_mask.shape

        # --- 1. Structural closure ---
        active_in_degree = torch.bmm(
            adjacency.transpose(1, 2),
            active_mask.unsqueeze(-1)).squeeze(-1)
        active_out_degree = torch.bmm(
            adjacency,
            active_mask.unsqueeze(-1)).squeeze(-1)

        has_active_parents = (active_in_degree * active_mask).sum(dim=-1, keepdim=True)
        has_active_children = (active_out_degree * active_mask).sum(dim=-1, keepdim=True)

        in_deg = adjacency.sum(dim=1)
        out_deg = adjacency.sum(dim=2)
        root_nodes = (in_deg < 0.5).float()
        terminal_nodes = (out_deg < 0.5).float()
        roots_covered = (active_mask * root_nodes).sum(dim=-1, keepdim=True)
        terminals_covered = (active_mask * terminal_nodes).sum(dim=-1, keepdim=True)
        root_total = root_nodes.sum(dim=-1, keepdim=True).clamp(min=1)
        terminal_total = terminal_nodes.sum(dim=-1, keepdim=True).clamp(min=1)

        structural_features = torch.cat([
            roots_covered / root_total,
            terminals_covered / terminal_total,
            active_mask.sum(dim=-1, keepdim=True) / N,
        ], dim=-1)
        structural_score = torch.sigmoid(self.structural_net(structural_features))

        # --- 2. Predictive adequacy ---
        residual = (observed - predicted).abs()
        masked_residual = (residual * active_mask).sum(dim=-1, keepdim=True) / active_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        predictive_features = torch.cat([residual, sufficiency], dim=-1)
        predictive_score = torch.sigmoid(self.predictive_net(predictive_features))

        # --- 3. Marginal diminishment ---
        if prev_predicted is not None:
            prediction_delta = (predicted - prev_predicted).abs().mean(dim=-1, keepdim=True)
        else:
            prediction_delta = torch.ones(B, 1, device=active_mask.device)
        marginal_features = torch.cat([prediction_delta, masked_residual], dim=-1)
        marginal_score = torch.sigmoid(self.marginal_net(marginal_features))

        # --- Fusion ---
        all_scores = torch.cat([structural_score, predictive_score, marginal_score], dim=-1)
        halt_prob = self.fusion(all_scores)

        details = {
            "structural": structural_score,
            "predictive": predictive_score,
            "marginal": marginal_score,
        }
        return halt_prob, details


class SparseReasoningEngine(nn.Module):
    """
    The full sparse reasoning loop with adaptive halting.

    Uses Adaptive Computation Time (ACT): each step produces a
    halting probability h_t. Accumulate sum(h_t) across steps;
    halt at the step where cumulative sum first exceeds 1.
    The final output is a weighted combination of per-step
    predictions, where the weights come from the halting
    distribution — so the model learns to allocate more "thought"
    to harder problems.

    Simple problem:  2 steps, h=[0.3, 0.8] → stops at step 2.
    Complex problem: 5 steps, h=[0.1, 0.15, 0.2, 0.25, 0.35] → uses all.
    """

    def __init__(self, num_nodes: int, query_dim: int,
                 initial_k: int = 5, max_steps: int = 6,
                 hidden_dim: int = 64):
        super().__init__()
        self.num_nodes = num_nodes
        self.initial_k = min(initial_k, num_nodes)
        self.max_steps = max_steps

        self.causal_graph = DifferentiableCausalGraph(num_nodes)
        self.relevance_scorer = RelevanceScorer(num_nodes, query_dim, hidden_dim)
        self.sufficiency_detector = SufficiencyDetector(num_nodes, hidden_dim)
        self.expansion_strategy = ExpansionStrategy(num_nodes, hidden_dim)
        self.convergence_detector = ConvergenceDetector(num_nodes, hidden_dim)

    def _masked_forward(self, x: torch.Tensor, adj: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        B, N = x.shape
        masked_x = x * mask
        masked_adj = adj * mask.unsqueeze(1) * mask.unsqueeze(2)

        in_degrees = masked_adj.mean(0).sum(dim=0)
        order = torch.argsort(in_degrees).tolist()

        result = torch.zeros_like(x)
        for idx in order:
            parent_values = masked_x * masked_adj[:, :, idx]
            result[:, idx] = self.causal_graph.structural_eqs[idx](
                parent_values).squeeze(-1)
        return result * mask

    def forward(self, query: torch.Tensor, observations: torch.Tensor
                ) -> dict[str, torch.Tensor]:
        B, N = observations.shape
        adj = self.causal_graph.adjacency.unsqueeze(0).expand(B, -1, -1)

        # --- Phase 1: Relevance scoring and initial activation ---
        relevance = self.relevance_scorer(query, observations, adj)

        if self.initial_k < N:
            topk_vals, topk_idx = relevance.topk(self.initial_k, dim=-1)
            hard_mask = torch.zeros_like(relevance)
            hard_mask.scatter_(1, topk_idx, 1.0)
            active_mask = hard_mask * relevance + (1 - hard_mask) * 0
        else:
            active_mask = relevance

        # --- Phase 2: Adaptive reasoning loop ---
        step_predictions = []
        step_masks = [active_mask]
        step_sufficiency = []
        step_directions = []
        step_insufficiency = []
        step_halt_probs = []
        step_convergence_details = []
        nodes_activated = [active_mask.sum(dim=-1).mean()]

        cumulative_halt = torch.zeros(B, 1, device=observations.device)
        remainders = torch.ones(B, 1, device=observations.device)
        halting_weights = []
        prev_pred = None
        actual_steps = self.max_steps

        for step in range(self.max_steps):
            pred = self._masked_forward(observations, adj, active_mask)
            step_predictions.append(pred)

            sufficient, insufficiency = self.sufficiency_detector(
                pred, observations, active_mask)
            step_sufficiency.append(sufficient)
            step_insufficiency.append(insufficiency)

            halt_prob, conv_details = self.convergence_detector(
                active_mask, adj, pred, observations, prev_pred, sufficient)
            step_halt_probs.append(halt_prob)
            step_convergence_details.append(conv_details)

            still_running = (cumulative_halt < 1.0).float()
            new_cumulative = cumulative_halt + halt_prob * still_running
            overshoot = (new_cumulative - 1.0).clamp(min=0)
            effective_halt = halt_prob * still_running - overshoot

            halting_weights.append(effective_halt)
            cumulative_halt = new_cumulative.clamp(max=1.0)
            remainders = (1.0 - cumulative_halt).clamp(min=0)

            if step < self.max_steps - 1:
                if remainders.mean().item() > 0.01:
                    active_mask, direction = self.expansion_strategy(
                        insufficiency, active_mask, adj, observations)
                    step_directions.append(direction)
                else:
                    step_directions.append(
                        torch.zeros(B, 2, device=observations.device))
                    actual_steps = step + 1

                step_masks.append(active_mask)
                nodes_activated.append(active_mask.sum(dim=-1).mean())

            prev_pred = pred

            if actual_steps <= step + 1:
                break

        if remainders.sum() > 0:
            halting_weights.append(remainders)
            if len(halting_weights) > len(step_predictions):
                halting_weights = halting_weights[:len(step_predictions)]

        total_weight = sum(w.abs() for w in halting_weights)
        total_weight = total_weight.clamp(min=1e-8)
        normalized_weights = [w / total_weight for w in halting_weights]

        weighted_prediction = sum(
            w * p for w, p in zip(normalized_weights, step_predictions))

        mean_adj = adj.mean(dim=0)
        d = N
        M = torch.eye(d, device=mean_adj.device) + mean_adj / d
        E = torch.matrix_power(M, d)
        dag_penalty = torch.trace(E) - d

        ponder_cost = sum(
            (1 - h.squeeze(-1)) for h in step_halt_probs).mean()

        return {
            "predictions": step_predictions,
            "weighted_prediction": weighted_prediction,
            "active_masks": step_masks,
            "sufficiency": step_sufficiency,
            "insufficiency": step_insufficiency,
            "directions": step_directions,
            "halt_probs": step_halt_probs,
            "halting_weights": normalized_weights,
            "convergence_details": step_convergence_details,
            "relevance": relevance,
            "dag_penalty": dag_penalty,
            "nodes_activated": nodes_activated,
            "ponder_cost": ponder_cost,
            "actual_steps": len(step_predictions),
            "final_adj": adj,
        }

    def loss(self, output: dict, true_adj: torch.Tensor,
             observations: torch.Tensor, query: torch.Tensor,
             weights: Optional[dict[str, float]] = None
             ) -> dict[str, torch.Tensor]:
        w = {
            "prediction": 1.0,
            "adjacency": 1.0,
            "dag": 0.1,
            "sparsity": 0.05,
            "efficiency": 0.2,
            "ponder": 0.01,
        }
        if weights:
            w.update(weights)

        device = observations.device

        pred_loss = F.mse_loss(output["weighted_prediction"], observations)

        final_adj = output["final_adj"]
        true_adj_exp = true_adj.unsqueeze(0).expand_as(final_adj)
        adj_loss = F.binary_cross_entropy(
            final_adj.clamp(1e-6, 1 - 1e-6), true_adj_exp)

        dag_loss = output["dag_penalty"]
        sparsity_loss = final_adj.mean()

        total_activated = output["nodes_activated"][-1]
        efficiency_loss = total_activated / self.num_nodes

        ponder_loss = output["ponder_cost"]

        total = (w["prediction"] * pred_loss
                 + w["adjacency"] * adj_loss
                 + w["dag"] * dag_loss
                 + w["sparsity"] * sparsity_loss
                 + w["efficiency"] * efficiency_loss
                 + w["ponder"] * ponder_loss)

        return {
            "total": total,
            "prediction": pred_loss,
            "adjacency": adj_loss,
            "dag": dag_loss,
            "sparsity": sparsity_loss,
            "efficiency": efficiency_loss,
            "ponder": ponder_loss,
        }
