"""
Path 3: Hypothesis-Verification Loop.

Transformer generates causal hypotheses (candidate edges),
Causal graph engine verifies them via do-calculus,
Verification result serves as reward signal.
"""

import torch
import torch.nn as nn
from typing import Optional

from crt.graph.causal_graph import DifferentiableCausalGraph


class HypothesisGenerator(nn.Module):
    """Transformer that proposes candidate causal edges given observations."""

    def __init__(self, num_nodes: int, d_model: int = 128, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_embedding = nn.Embedding(num_nodes, d_model)
        self.observation_proj = nn.Linear(num_nodes, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_model * 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output: for each pair (i,j), predict P(i causes j).
        self.edge_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, 1), nn.Sigmoid(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: (batch, num_nodes) observed values.
        Returns:
            edge_probs: (batch, num_nodes, num_nodes) predicted edge probabilities.
        """
        B = observations.shape[0]
        node_ids = torch.arange(self.num_nodes, device=observations.device)
        node_emb = self.node_embedding(node_ids).unsqueeze(0).expand(B, -1, -1)
        obs_emb = self.observation_proj(observations).unsqueeze(1).expand(-1, self.num_nodes, -1)
        x = node_emb + obs_emb
        x = self.transformer(x)

        # Pairwise edge prediction.
        xi = x.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
        xj = x.unsqueeze(1).expand(-1, self.num_nodes, -1, -1)
        pairs = torch.cat([xi, xj], dim=-1)
        edge_probs = self.edge_predictor(pairs).squeeze(-1)

        # Zero diagonal.
        mask = 1 - torch.eye(self.num_nodes, device=edge_probs.device)
        return edge_probs * mask.unsqueeze(0)


class CausalVerifier(nn.Module):
    """
    Verifies causal hypotheses by checking interventional consistency.

    Given a proposed graph and observational data, performs interventions
    and checks if the predicted outcomes match observations.
    """

    def __init__(self, num_nodes: int):
        super().__init__()
        self.causal_graph = DifferentiableCausalGraph(num_nodes)

    def verify(self, proposed_adj: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute a verification score for the proposed adjacency matrix.

        For each node, simulate do(X_i = observed_i) and check if
        downstream predictions match observations.

        Returns:
            score: (batch,) verification score in [0, 1]. Higher = more consistent.
        """
        B, N = observations.shape
        self.causal_graph.edge_logits.data = torch.log(proposed_adj / (1 - proposed_adj + 1e-8) + 1e-8).mean(0)

        total_consistency = torch.zeros(B, device=observations.device)
        for i in range(N):
            intervened = self.causal_graph.intervene(i, observations[:, i], observations)
            consistency = 1 - (intervened - observations).pow(2).mean(dim=-1)
            total_consistency += consistency

        return total_consistency / N


class HypothesisVerifier(nn.Module):
    """
    Path 3 complete model.

    Loop:
    1. Generator proposes causal graph from observations.
    2. Verifier checks interventional consistency.
    3. Verification score = reward for generator.
    """

    def __init__(self, num_nodes: int, d_model: int = 128):
        super().__init__()
        self.generator = HypothesisGenerator(num_nodes, d_model)
        self.verifier = CausalVerifier(num_nodes)

    def forward(self, observations: torch.Tensor) -> dict:
        proposed_adj = self.generator(observations)
        verification_score = self.verifier.verify(proposed_adj, observations)

        return {
            "proposed_adj": proposed_adj,
            "verification_score": verification_score,
            "dag_penalty": self.verifier.causal_graph.dag_penalty(),
        }

    def loss(self, output: dict, true_adj: Optional[torch.Tensor] = None,
             dag_weight: float = 0.1, verify_weight: float = 0.5) -> torch.Tensor:
        # Maximize verification score.
        verify_loss = -output["verification_score"].mean()

        # Supervised loss on adjacency if ground truth is available.
        sup_loss = torch.tensor(0.0, device=verify_loss.device)
        if true_adj is not None:
            sup_loss = nn.functional.binary_cross_entropy(
                output["proposed_adj"], true_adj.unsqueeze(0).expand_as(output["proposed_adj"]))

        # DAG constraint.
        dag_loss = output["dag_penalty"]

        return sup_loss + verify_weight * verify_loss + dag_weight * dag_loss
