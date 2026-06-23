"""
Path 1: Perception-Reasoning Separation.

Transformer handles perception (encoding unstructured inputs),
Causal Graph handles reasoning (structured inference).
"""

import torch
import torch.nn as nn
from typing import Optional

from crt.graph.causal_graph import DifferentiableCausalGraph
from crt.transformer.causal_attention import CausalTransformerBlock


class PerceptionEncoder(nn.Module):
    """Transformer encoder that converts raw observations into node state vectors."""

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, num_nodes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        self.layers = nn.ModuleList([
            CausalTransformerBlock(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])
        self.node_projection = nn.Linear(d_model, num_nodes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token IDs (e.g., log text).
        Returns:
            node_states: (batch, num_nodes) activation level per causal node.
        """
        x = self.embedding(input_ids) + self.pos_encoding[:, :input_ids.size(1)]
        for layer in self.layers:
            x = layer(x)
        pooled = x.mean(dim=1)
        return torch.sigmoid(self.node_projection(pooled))


class CausalReasoner(nn.Module):
    """Reasons over node states using the causal graph to produce a diagnosis."""

    def __init__(self, num_nodes: int, hidden_dim: int = 64):
        super().__init__()
        self.causal_graph = DifferentiableCausalGraph(num_nodes)
        self.classifier = nn.Sequential(
            nn.Linear(num_nodes * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_nodes),
        )

    def forward(self, node_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_states: (batch, num_nodes) observed activation levels.
        Returns:
            root_cause_logits: (batch, num_nodes) probability of each node being the root cause.
            dag_penalty: scalar for DAG constraint loss.
        """
        propagated = self.causal_graph(node_states)
        combined = torch.cat([node_states, propagated], dim=-1)
        root_cause_logits = self.classifier(combined)
        return root_cause_logits, self.causal_graph.dag_penalty()


class PerceptionReasoningModel(nn.Module):
    """
    Path 1 complete model: Perception + Reasoning.

    Input: raw text (logs, metrics) -> Transformer encodes -> node states
    -> Causal graph reasons -> root cause prediction.
    """

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, num_causal_nodes: int):
        super().__init__()
        self.encoder = PerceptionEncoder(vocab_size, d_model, num_heads, num_layers, num_causal_nodes)
        self.reasoner = CausalReasoner(num_causal_nodes)

    def forward(self, input_ids: torch.Tensor) -> dict:
        node_states = self.encoder(input_ids)
        root_cause_logits, dag_penalty = self.reasoner(node_states)
        return {
            "node_states": node_states,
            "root_cause_logits": root_cause_logits,
            "dag_penalty": dag_penalty,
        }

    def loss(self, output: dict, targets: torch.Tensor, dag_weight: float = 0.1) -> torch.Tensor:
        ce = nn.functional.cross_entropy(output["root_cause_logits"], targets)
        return ce + dag_weight * output["dag_penalty"]
