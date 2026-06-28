"""
Hierarchical Causal Graph — multi-level causal reasoning.

Supports:
- Multiple abstraction levels (micro → meso → macro)
- Intra-level causal edges (same level)
- Cross-level aggregation (bottom-up: micro details → macro concepts)
- Cross-level grounding (top-down: macro context → micro predictions)
- Level-aware structural equations
"""

import torch
import torch.nn as nn
from typing import Optional

from crt.graph.causal_graph import DifferentiableCausalGraph


class CrossLevelBridge(nn.Module):
    """Learns aggregation (bottom-up) and grounding (top-down) between two levels."""

    def __init__(self, lower_nodes: int, upper_nodes: int, hidden_dim: int = 64):
        super().__init__()
        self.aggregate = nn.Sequential(
            nn.Linear(lower_nodes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, upper_nodes),
        )
        self.ground = nn.Sequential(
            nn.Linear(upper_nodes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, lower_nodes),
        )
        self.agg_gate = nn.Sequential(
            nn.Linear(lower_nodes + upper_nodes, 1),
            nn.Sigmoid(),
        )

    def bottom_up(self, lower_states: torch.Tensor,
                  upper_states: torch.Tensor) -> torch.Tensor:
        aggregated = self.aggregate(lower_states)
        gate_input = torch.cat([lower_states, upper_states], dim=-1)
        gate = self.agg_gate(gate_input)
        return upper_states + gate * aggregated

    def top_down(self, upper_states: torch.Tensor,
                 lower_states: torch.Tensor) -> torch.Tensor:
        grounded = self.ground(upper_states)
        return lower_states + torch.sigmoid(grounded) * lower_states


class HierarchicalCausalGraph(nn.Module):
    """
    Multi-level causal graph where each level has its own
    DifferentiableCausalGraph and levels are connected by
    learnable cross-level bridges.

    Example with 3 levels:
        Level 0 (micro):  8 nodes — raw observables
        Level 1 (meso):   4 nodes — intermediate concepts
        Level 2 (macro):  2 nodes — high-level causes

    Reasoning flows both bottom-up (aggregate evidence) and
    top-down (propagate context), enabling "zooming" between
    abstraction levels during the revision loop.
    """

    def __init__(self, level_sizes: list[int], hidden_dim: int = 64):
        super().__init__()
        self.level_sizes = level_sizes
        self.num_levels = len(level_sizes)

        self.levels = nn.ModuleList([
            DifferentiableCausalGraph(n) for n in level_sizes
        ])

        self.bridges = nn.ModuleList()
        for i in range(self.num_levels - 1):
            self.bridges.append(
                CrossLevelBridge(level_sizes[i], level_sizes[i + 1], hidden_dim))

    @property
    def total_nodes(self) -> int:
        return sum(self.level_sizes)

    def adjacency_per_level(self) -> list[torch.Tensor]:
        return [level.adjacency for level in self.levels]

    def full_adjacency(self) -> torch.Tensor:
        """Build a block-diagonal adjacency from all levels."""
        N = self.total_nodes
        device = self.levels[0].edge_logits.device
        full = torch.zeros(N, N, device=device)
        offset = 0
        for level in self.levels:
            n = level.num_nodes
            full[offset:offset + n, offset:offset + n] = level.adjacency
            offset += n
        return full

    def dag_penalty(self) -> torch.Tensor:
        return sum(level.dag_penalty() for level in self.levels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass with bottom-up then top-down message passing.

        Args:
            x: (B, total_nodes) input with all levels concatenated.
        Returns:
            dict with "output", "level_states", "level_adjs".
        """
        level_inputs = self._split_by_level(x)
        level_states = []
        for i, (level, inp) in enumerate(zip(self.levels, level_inputs)):
            level_states.append(level(inp))

        for i in range(self.num_levels - 1):
            level_states[i + 1] = self.bridges[i].bottom_up(
                level_states[i], level_states[i + 1])

        for i in range(self.num_levels - 2, -1, -1):
            level_states[i] = self.bridges[i].top_down(
                level_states[i + 1], level_states[i])

        output = torch.cat(level_states, dim=-1)
        return {
            "output": output,
            "level_states": level_states,
            "level_adjs": self.adjacency_per_level(),
        }

    def forward_at_level(self, x: torch.Tensor,
                         level_idx: int) -> torch.Tensor:
        return self.levels[level_idx](x)

    def _split_by_level(self, x: torch.Tensor) -> list[torch.Tensor]:
        parts = []
        offset = 0
        for size in self.level_sizes:
            parts.append(x[:, offset:offset + size])
            offset += size
        return parts

    def to_flat_index(self, level_idx: int, node_idx: int) -> int:
        return sum(self.level_sizes[:level_idx]) + node_idx
