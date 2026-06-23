"""
Differentiable Causal Graph Engine.

Supports:
- Standard causal graph operations (add nodes/edges, trace upstream/downstream)
- Differentiable adjacency matrix for end-to-end training
- do-calculus interventions as differentiable operations
- Structural Causal Model (SCM) with learnable structural equations
"""

import torch
import torch.nn as nn
import networkx as nx
from typing import Optional


class CausalGraph:
    """Standard causal graph with discrete operations."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node_id: str, **attrs):
        self.graph.add_node(node_id, **attrs)

    def add_edge(self, cause: str, effect: str, confidence: float = 1.0):
        self.graph.add_edge(cause, effect, confidence=confidence)

    def causes_of(self, node_id: str) -> list[str]:
        return list(self.graph.predecessors(node_id))

    def effects_of(self, node_id: str) -> list[str]:
        return list(self.graph.successors(node_id))

    def trace_upstream(self, node_id: str) -> list[str]:
        return list(nx.ancestors(self.graph, node_id))

    def trace_downstream(self, node_id: str) -> list[str]:
        return list(nx.descendants(self.graph, node_id))

    def is_dag(self) -> bool:
        return nx.is_directed_acyclic_graph(self.graph)

    def topological_order(self) -> list[str]:
        return list(nx.topological_sort(self.graph))


class DifferentiableCausalGraph(nn.Module):
    """
    A causal graph with a learnable adjacency matrix.

    The adjacency matrix is parameterized as continuous values in [0,1],
    allowing gradient-based optimization of graph structure.
    DAG constraint is enforced via a differentiable acyclicity penalty
    (NOTEARS: Zheng et al., 2018).
    """

    def __init__(self, num_nodes: int, node_names: Optional[list[str]] = None):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_names = node_names or [f"x_{i}" for i in range(num_nodes)]

        # Learnable edge weights (logits before sigmoid).
        self.edge_logits = nn.Parameter(torch.zeros(num_nodes, num_nodes))

        # Structural equations: each node is a function of its parents.
        self.structural_eqs = nn.ModuleList([
            nn.Sequential(nn.Linear(num_nodes, 64), nn.ReLU(), nn.Linear(64, 1))
            for _ in range(num_nodes)
        ])

    @property
    def adjacency(self) -> torch.Tensor:
        """Soft adjacency matrix in [0, 1]."""
        adj = torch.sigmoid(self.edge_logits)
        # Zero out diagonal (no self-loops).
        return adj * (1 - torch.eye(self.num_nodes, device=adj.device))

    def dag_penalty(self) -> torch.Tensor:
        """
        Differentiable DAG constraint (NOTEARS).
        Returns 0 iff the graph is a DAG.
        h(A) = tr(e^A) - d = 0 iff A is a DAG.
        """
        adj = self.adjacency
        d = self.num_nodes
        M = torch.eye(d, device=adj.device) + adj / d
        # Matrix power approximation for efficiency.
        E = torch.matrix_power(M, d)
        return torch.trace(E) - d

    def intervene(self, node_idx: int, value: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        """
        do(X_i = value): simulate an intervention by cutting incoming edges
        to node_idx and setting its value, then propagating forward.
        """
        adj = self.adjacency.clone()
        # Cut incoming edges to the intervention target.
        adj[:, node_idx] = 0

        result = observations.clone()
        result[:, node_idx] = value

        # Forward propagate through topological order.
        order = self._approx_topological_order(adj)
        for idx in order:
            if idx == node_idx:
                continue
            parent_values = result * adj[:, idx].unsqueeze(0)
            result[:, idx] = self.structural_eqs[idx](parent_values).squeeze(-1)

        return result

    def counterfactual(self, observation: torch.Tensor, node_idx: int, new_value: torch.Tensor) -> torch.Tensor:
        """
        Counterfactual: 'What would Y have been if X had been new_value?'
        Three steps: abduction, intervention, prediction.
        """
        # Step 1: Abduction - infer exogenous noise from observation.
        # (Simplified: treat residuals as noise)
        predicted = self.forward(observation)
        noise = observation - predicted

        # Step 2: Intervention.
        intervened = self.intervene(node_idx, new_value, observation)

        # Step 3: Prediction with noise added back.
        return intervened + noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through structural equations following causal order."""
        adj = self.adjacency
        result = torch.zeros_like(x)
        order = self._approx_topological_order(adj)
        for idx in order:
            parent_values = x * adj[:, idx].unsqueeze(0)
            result[:, idx] = self.structural_eqs[idx](parent_values).squeeze(-1)
        return result

    def _approx_topological_order(self, adj: torch.Tensor) -> list[int]:
        """Approximate topological order from soft adjacency."""
        # Use in-degree as proxy for ordering.
        in_degrees = adj.sum(dim=0)
        return torch.argsort(in_degrees).tolist()

    def to_discrete(self, threshold: float = 0.5) -> CausalGraph:
        """Convert soft adjacency to a discrete CausalGraph."""
        g = CausalGraph()
        adj = self.adjacency.detach().cpu()
        for i, name in enumerate(self.node_names):
            g.add_node(name)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if adj[i, j] > threshold:
                    g.add_edge(self.node_names[i], self.node_names[j],
                               confidence=adj[i, j].item())
        return g
