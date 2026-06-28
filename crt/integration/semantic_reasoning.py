"""
Semantic Causal Reasoning — bridging language and causal graphs.

The gap between CRT and real-world reasoning:
  Numbers  → causal graph → numbers     (what CRT does now)
  Language → causal graph → explanation  (what this module enables)

Architecture:

  "Patient has fever, cough, got wet in rain 3 days ago"
      ↓
  [SemanticEncoder] — extracts which causal variables are mentioned
      ↓                and estimates their activation level
  (node_activations: "rain"=0.9, "cold"=0.7, "fever"=0.8, ...)
      ↓
  [SparseReasoningEngine] — reasons over the causal graph
      ↓
  (causal_chain: rain → cold → fever)
      ↓
  [ExplanationDecoder] — converts causal chain back to language
      ↓
  "Getting wet in rain likely caused a cold, leading to the fever"

Two modes of connecting language to causal nodes:
  1. Predefined schema: node names are known, text is matched to them.
     Suitable for domain-specific applications (medical, legal, ops).
  2. Embedding-based: node semantics are learned from descriptions.
     Suitable for open-domain or when schemas evolve.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SemanticEncoder(nn.Module):
    """
    Converts text tokens into causal node activations.

    Each causal node has a learnable semantic description vector.
    The encoder computes soft attention between input tokens and
    node descriptions to produce per-node activation levels.

    This is the bridge: language → (B, num_nodes) activation vector
    that the causal graph engine can reason over.
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_nodes: int,
                 num_heads: int = 4, num_layers: int = 2,
                 max_seq_len: int = 512):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, embed_dim * 4, batch_first=True)
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.node_descriptions = nn.Parameter(
            torch.randn(num_nodes, embed_dim) * 0.02)

        self.activation_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids:      (B, seq_len) token IDs.
            attention_mask: (B, seq_len) 1=real token, 0=padding.
        Returns:
            activations:    (B, num_nodes) in [0, 1].
            node_attention: (B, num_nodes, seq_len) which tokens
                            activated which nodes (interpretable).
        """
        B, L = input_ids.shape

        x = self.token_embedding(input_ids) + self.pos_embedding[:, :L]

        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        text_features = self.text_encoder(
            x, src_key_padding_mask=src_key_padding_mask)

        node_desc = self.node_descriptions.unsqueeze(0).expand(B, -1, -1)
        attn_scores = torch.bmm(
            node_desc, text_features.transpose(1, 2)) / (self.embed_dim ** 0.5)

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(1).expand(-1, self.num_nodes, -1)
            attn_scores = attn_scores.masked_fill(~mask_expanded.bool(), float('-inf'))

        node_attention = F.softmax(attn_scores, dim=-1)

        node_context = torch.bmm(node_attention, text_features)

        activations = self.activation_head(node_context).squeeze(-1)
        activations = torch.sigmoid(activations)

        return activations, node_attention


class QueryEncoder(nn.Module):
    """
    Encodes a natural language query into the query vector
    expected by SparseReasoningEngine.

    "Why does the patient have a fever?"
        → (B, query_dim) dense vector capturing the question intent.
    """

    def __init__(self, vocab_size: int, embed_dim: int, query_dim: int,
                 num_heads: int = 4, num_layers: int = 2,
                 max_seq_len: int = 128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, embed_dim * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.pool_proj = nn.Linear(embed_dim, query_dim)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, seq_len) query token IDs.
            attention_mask: (B, seq_len) mask.
        Returns:
            query_vector: (B, query_dim).
        """
        B, L = input_ids.shape
        x = self.token_embedding(input_ids) + self.pos_embedding[:, :L]

        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        features = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        if attention_mask is not None:
            mask_f = attention_mask.unsqueeze(-1).float()
            pooled = (features * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            pooled = features.mean(dim=1)

        return self.pool_proj(pooled)


class ExplanationDecoder(nn.Module):
    """
    Converts causal reasoning results back into interpretable output.

    Takes the active node mask, causal adjacency, and node descriptions
    to produce:
    1. A causal chain (ordered list of activated nodes)
    2. Per-edge confidence scores
    3. A summary embedding that can be decoded to text
    """

    def __init__(self, num_nodes: int, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.num_nodes = num_nodes
        self.chain_scorer = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.summary_net = nn.Sequential(
            nn.Linear(num_nodes + num_nodes * num_nodes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, active_mask: torch.Tensor, adjacency: torch.Tensor,
                node_descriptions: torch.Tensor,
                node_activations: torch.Tensor
                ) -> dict[str, torch.Tensor]:
        """
        Args:
            active_mask:       (B, N) which nodes are active.
            adjacency:         (B, N, N) discovered causal adjacency.
            node_descriptions: (N, embed_dim) semantic embeddings of nodes.
            node_activations:  (B, N) activation levels.
        Returns:
            dict with "chain_scores", "edge_confidences", "summary".
        """
        B, N = active_mask.shape
        desc = node_descriptions.unsqueeze(0).expand(B, -1, -1)

        desc_i = desc.unsqueeze(2).expand(-1, -1, N, -1)
        desc_j = desc.unsqueeze(1).expand(-1, N, -1, -1)
        edge_w = adjacency.unsqueeze(-1)
        pair_features = torch.cat([desc_i, desc_j, edge_w], dim=-1)
        edge_confidences = self.chain_scorer(pair_features).squeeze(-1)
        edge_confidences = edge_confidences * adjacency

        active_adj = adjacency * active_mask.unsqueeze(1) * active_mask.unsqueeze(2)
        chain_scores = active_adj * edge_confidences

        flat_adj = adjacency.view(B, -1)
        summary_input = torch.cat([node_activations, flat_adj], dim=-1)
        summary = self.summary_net(summary_input)

        return {
            "chain_scores": chain_scores,
            "edge_confidences": edge_confidences,
            "summary": summary,
        }


class SemanticCausalModel(nn.Module):
    """
    End-to-end model: Text → Causal Reasoning → Explanation.

    Connects all components:
    1. SemanticEncoder: text → node activations
    2. QueryEncoder: question → query vector
    3. SparseReasoningEngine: activations + query → causal chain
    4. ExplanationDecoder: causal chain → interpretable output
    """

    def __init__(self, vocab_size: int, num_nodes: int,
                 embed_dim: int = 128, query_dim: int = 64,
                 initial_k: int = 5, max_steps: int = 4,
                 hidden_dim: int = 64):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim

        self.semantic_encoder = SemanticEncoder(
            vocab_size, embed_dim, num_nodes)
        self.query_encoder = QueryEncoder(
            vocab_size, embed_dim, query_dim)

        from crt.integration.sparse_reasoning import SparseReasoningEngine
        self.reasoning_engine = SparseReasoningEngine(
            num_nodes, query_dim, initial_k, max_steps, hidden_dim)

        self.explanation_decoder = ExplanationDecoder(
            num_nodes, embed_dim, hidden_dim)

    def forward(self, context_ids: torch.Tensor,
                query_ids: torch.Tensor,
                context_mask: Optional[torch.Tensor] = None,
                query_mask: Optional[torch.Tensor] = None
                ) -> dict[str, torch.Tensor]:
        """
        Args:
            context_ids:  (B, ctx_len) token IDs of the context/evidence.
            query_ids:    (B, q_len)   token IDs of the question.
            context_mask: (B, ctx_len) attention mask.
            query_mask:   (B, q_len)   attention mask.
        Returns:
            Full reasoning output including explanations.
        """
        node_activations, node_attention = self.semantic_encoder(
            context_ids, context_mask)

        query_vector = self.query_encoder(query_ids, query_mask)

        reasoning_output = self.reasoning_engine(query_vector, node_activations)

        final_mask = reasoning_output["active_masks"][-1]
        final_adj = reasoning_output["final_adj"]
        explanation = self.explanation_decoder(
            final_mask, final_adj.mean(0).unsqueeze(0).expand(final_mask.shape[0], -1, -1),
            self.semantic_encoder.node_descriptions,
            node_activations)

        return {
            **reasoning_output,
            "node_activations": node_activations,
            "node_attention": node_attention,
            "explanation": explanation,
        }

    def loss(self, output: dict, true_adj: torch.Tensor,
             target_nodes: Optional[torch.Tensor] = None,
             weights: Optional[dict[str, float]] = None
             ) -> dict[str, torch.Tensor]:
        """
        Args:
            output:       forward() output dict.
            true_adj:     (N, N) ground-truth causal adjacency.
            target_nodes: (B, N) optional ground-truth node activations.
            weights:      loss component weights.
        """
        w = {
            "prediction": 1.0,
            "adjacency": 1.0,
            "dag": 0.1,
            "sparsity": 0.05,
            "efficiency": 0.2,
            "ponder": 0.01,
            "grounding": 0.5,
        }
        if weights:
            w.update(weights)

        device = output["node_activations"].device
        obs = output["node_activations"]

        base_losses = self.reasoning_engine.loss(
            output, true_adj, obs,
            torch.zeros(obs.shape[0], self.reasoning_engine.relevance_scorer.query_proj.in_features, device=device))

        grounding_loss = torch.tensor(0.0, device=device)
        if target_nodes is not None:
            grounding_loss = F.mse_loss(output["node_activations"], target_nodes)

        total = base_losses["total"] + w["grounding"] * grounding_loss

        return {
            **base_losses,
            "total": total,
            "grounding": grounding_loss,
        }


class CausalTextDataset(torch.utils.data.Dataset):
    """
    Dataset for semantic causal reasoning.

    Each item has:
        context_ids:     token IDs describing a situation
        query_ids:       token IDs of a causal question
        context_mask:    attention mask for context
        query_mask:      attention mask for query
        adjacency:       (N, N) ground-truth causal graph
        target_nodes:    (N,) ground-truth node activations
        node_names:      list of node name strings
        explanation:     ground-truth causal chain as text
    """

    def __init__(self, node_names: list[str],
                 adjacency: torch.Tensor,
                 scenarios: list[dict],
                 tokenizer_fn=None,
                 max_context_len: int = 128,
                 max_query_len: int = 64):
        """
        Args:
            node_names:  list of causal variable names.
            adjacency:   (N, N) causal graph.
            scenarios:    list of dicts with keys:
                          "context": str, "query": str,
                          "target_nodes": list[float],
                          "explanation": str (optional).
            tokenizer_fn: callable(text, max_len) → (ids, mask) tensors.
                          If None, uses character-level tokenization.
            max_context_len: max token length for context.
            max_query_len:   max token length for query.
        """
        self.node_names = node_names
        self.adjacency = adjacency
        self.scenarios = scenarios
        self.max_context_len = max_context_len
        self.max_query_len = max_query_len

        if tokenizer_fn is not None:
            self.tokenizer_fn = tokenizer_fn
        else:
            self.tokenizer_fn = self._char_tokenize

    def _char_tokenize(self, text: str, max_len: int
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [min(ord(c), 255) for c in text[:max_len]]
        pad_len = max_len - len(ids)
        mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [0] * pad_len
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.scenarios)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.scenarios[idx]
        ctx_ids, ctx_mask = self.tokenizer_fn(s["context"], self.max_context_len)
        q_ids, q_mask = self.tokenizer_fn(s["query"], self.max_query_len)

        return {
            "context_ids": ctx_ids,
            "query_ids": q_ids,
            "context_mask": ctx_mask,
            "query_mask": q_mask,
            "adjacency": self.adjacency,
            "target_nodes": torch.tensor(s["target_nodes"], dtype=torch.float32),
        }


def build_demo_medical_dataset() -> CausalTextDataset:
    """
    Builds a small demo dataset for medical causal reasoning.

    Causal graph:
        rain → cold → fever
        rain → cold → cough
        bacteria → infection → fever
        smoking → cough
        infection → fatigue
    """
    node_names = ["rain", "cold", "fever", "cough",
                  "bacteria", "infection", "smoking", "fatigue"]
    N = len(node_names)
    adj = torch.zeros(N, N)
    edges = [
        (0, 1),  # rain → cold
        (1, 2),  # cold → fever
        (1, 3),  # cold → cough
        (4, 5),  # bacteria → infection
        (5, 2),  # infection → fever
        (6, 3),  # smoking → cough
        (5, 7),  # infection → fatigue
    ]
    for i, j in edges:
        adj[i, j] = 1.0

    scenarios = [
        {
            "context": "Patient reports fever and cough. Got caught in heavy rain three days ago. No known bacterial exposure.",
            "query": "What caused the fever?",
            "target_nodes": [0.9, 0.8, 0.9, 0.7, 0.1, 0.1, 0.0, 0.1],
        },
        {
            "context": "Patient has high fever and severe fatigue. Lab results show bacterial infection. No recent rain exposure.",
            "query": "What is the root cause of the fever?",
            "target_nodes": [0.0, 0.1, 0.9, 0.1, 0.9, 0.9, 0.0, 0.8],
        },
        {
            "context": "Patient coughs frequently but has no fever. Has been a smoker for 20 years.",
            "query": "Why does the patient cough?",
            "target_nodes": [0.0, 0.1, 0.0, 0.9, 0.0, 0.0, 0.9, 0.0],
        },
        {
            "context": "Patient caught cold after rain, now has fever, cough, and fatigue. Recent wound shows signs of infection.",
            "query": "What are all the causes of the symptoms?",
            "target_nodes": [0.9, 0.8, 0.9, 0.8, 0.7, 0.8, 0.0, 0.7],
        },
        {
            "context": "Healthy patient with no symptoms. Regular checkup.",
            "query": "Are there any health concerns?",
            "target_nodes": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    ]

    return CausalTextDataset(
        node_names=node_names,
        adjacency=adj,
        scenarios=scenarios,
    )
