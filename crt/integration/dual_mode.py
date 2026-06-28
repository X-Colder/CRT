"""
Dual-mode CRT — two ways to integrate causal reasoning with language models.

Mode 1: Tool Mode
    CRT as an external reasoning tool. LLM sends text, CRT returns
    a structured causal chain. CRT runs OUTSIDE the LLM.

    LLM → "analyze this" → [SemanticEncoder → SparseReasoning → ExplanationDecoder] → causal chain → LLM

Mode 2: Embedded Mode
    CRT injected INTO the Transformer. Causal adjacency modulates
    attention scores so the model "thinks causally" at every layer.
    The causal graph is discovered and refined DURING the forward pass.

    Token → [CausalTransformerLayer₁] → ... → [CausalTransformerLayerₙ] → Token
                   ↑ causal graph                    ↑ refined graph

Usage:
    model = create_crt(mode="tool",     ...)   # external tool
    model = create_crt(mode="embedded", ...)   # internal attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

from crt.graph.causal_graph import DifferentiableCausalGraph
from crt.transformer.causal_attention import CausalConstrainedAttention, CausalTransformerBlock
from crt.integration.sparse_reasoning import SparseReasoningEngine
from crt.integration.semantic_reasoning import (
    SemanticEncoder, QueryEncoder, ExplanationDecoder)


# ---------------------------------------------------------------------------
# Mode 1: Tool Mode — CRT as external reasoning service
# ---------------------------------------------------------------------------

class ToolModeCRT(nn.Module):
    """
    CRT as an external tool for LLMs.

    Input:  context text + query text (token IDs)
    Output: structured causal chain + node activations + explanation

    The LLM calls this module like a function, gets back a structured
    result, and incorporates it into its generation.
    """

    def __init__(self, vocab_size: int, num_nodes: int,
                 embed_dim: int = 128, query_dim: int = 64,
                 initial_k: int = 5, max_steps: int = 4,
                 hidden_dim: int = 64, node_names: Optional[list[str]] = None):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.node_names = node_names or [f"node_{i}" for i in range(num_nodes)]

        self.semantic_encoder = SemanticEncoder(
            vocab_size, embed_dim, num_nodes)
        self.query_encoder = QueryEncoder(
            vocab_size, embed_dim, query_dim)
        self.reasoning_engine = SparseReasoningEngine(
            num_nodes, query_dim, initial_k, max_steps, hidden_dim)
        self.explanation_decoder = ExplanationDecoder(
            num_nodes, embed_dim, hidden_dim)

    def forward(self, context_ids: torch.Tensor,
                query_ids: torch.Tensor,
                context_mask: Optional[torch.Tensor] = None,
                query_mask: Optional[torch.Tensor] = None
                ) -> dict[str, torch.Tensor]:
        node_activations, node_attention = self.semantic_encoder(
            context_ids, context_mask)
        query_vector = self.query_encoder(query_ids, query_mask)
        reasoning = self.reasoning_engine(query_vector, node_activations)

        final_mask = reasoning["active_masks"][-1]
        final_adj = reasoning["final_adj"].mean(0, keepdim=True).expand(
            final_mask.shape[0], -1, -1)
        explanation = self.explanation_decoder(
            final_mask, final_adj,
            self.semantic_encoder.node_descriptions,
            node_activations)

        return {
            **reasoning,
            "node_activations": node_activations,
            "node_attention": node_attention,
            "explanation": explanation,
        }

    def loss(self, output: dict, true_adj: torch.Tensor,
             target_nodes: Optional[torch.Tensor] = None,
             weights: Optional[dict[str, float]] = None
             ) -> dict[str, torch.Tensor]:
        w = {"grounding": 0.5}
        if weights:
            w.update(weights)

        device = output["node_activations"].device
        obs = output["node_activations"]
        dummy_query = torch.zeros(
            obs.shape[0],
            self.query_encoder.pool_proj.out_features,
            device=device)

        base = self.reasoning_engine.loss(output, true_adj, obs, dummy_query, weights)

        grounding = torch.tensor(0.0, device=device)
        if target_nodes is not None:
            grounding = F.mse_loss(obs, target_nodes)

        total = base["total"] + w["grounding"] * grounding
        return {**base, "total": total, "grounding": grounding}

    @torch.no_grad()
    def reason(self, context_ids: torch.Tensor,
               query_ids: torch.Tensor,
               context_mask: Optional[torch.Tensor] = None,
               query_mask: Optional[torch.Tensor] = None,
               threshold: float = 0.5) -> dict:
        """
        Inference-time reasoning. Returns human-readable results.
        """
        self.eval()
        output = self.forward(context_ids, query_ids, context_mask, query_mask)

        acts = output["node_activations"][0].cpu()
        active = output["active_masks"][-1][0].cpu()
        adj = output["final_adj"][0].cpu()
        chain = output["explanation"]["chain_scores"][0].cpu()

        active_nodes = []
        for i, (name, a, m) in enumerate(
                zip(self.node_names, acts, active)):
            if m > threshold:
                active_nodes.append({
                    "name": name, "activation": a.item(),
                    "active_weight": m.item()})

        causal_edges = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if adj[i, j] > threshold and active[i] > threshold and active[j] > threshold:
                    causal_edges.append({
                        "cause": self.node_names[i],
                        "effect": self.node_names[j],
                        "strength": adj[i, j].item(),
                        "chain_confidence": chain[i, j].item(),
                    })

        convergence = output["convergence_details"][-1]
        return {
            "active_nodes": active_nodes,
            "causal_edges": causal_edges,
            "reasoning_steps": output["actual_steps"],
            "convergence": {
                "structural": convergence["structural"].item(),
                "predictive": convergence["predictive"].item(),
                "marginal": convergence["marginal"].item(),
            },
        }


# ---------------------------------------------------------------------------
# Mode 2: Embedded Mode — CRT injected into Transformer layers
# ---------------------------------------------------------------------------

class CausalGraphDiscoverer(nn.Module):
    """
    Discovers the causal adjacency matrix from token representations.
    Runs at each Transformer layer to refine the graph.
    """

    def __init__(self, d_model: int, num_nodes: int, hidden_dim: int = 64):
        super().__init__()
        self.num_nodes = num_nodes
        self.token_to_node = nn.Linear(d_model, num_nodes)
        self.edge_predictor = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_states: torch.Tensor,
                prev_adj: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        B = hidden_states.shape[0]
        N = self.num_nodes

        node_summary = self.token_to_node(hidden_states.mean(dim=1))
        node_i = node_summary.unsqueeze(2).expand(-1, -1, N)
        node_j = node_summary.unsqueeze(1).expand(-1, N, -1)
        pair_features = torch.stack([node_i, node_j], dim=-1)

        adj = torch.sigmoid(self.edge_predictor(pair_features).squeeze(-1))
        mask = 1 - torch.eye(N, device=adj.device)
        adj = adj * mask.unsqueeze(0)

        if prev_adj is not None:
            adj = 0.7 * adj + 0.3 * prev_adj

        return adj


class EmbeddedCausalLayer(nn.Module):
    """
    A single Transformer layer with embedded causal reasoning.

    Standard attention is modulated by the causal adjacency:
    tokens corresponding to causally related concepts attend
    more strongly to each other.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 num_nodes: int, dropout: float = 0.1,
                 causal_weight: float = 0.8, hidden_dim: int = 64):
        super().__init__()
        self.causal_attn = CausalConstrainedAttention(
            d_model, num_heads, causal_weight)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.graph_discoverer = CausalGraphDiscoverer(
            d_model, num_nodes, hidden_dim)

        self.token_node_mapper = nn.Linear(d_model, num_nodes)

    def forward(self, x: torch.Tensor,
                prev_adj: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x:   (B, seq_len, d_model) updated hidden states.
            adj: (B, N, N) discovered/refined causal adjacency.
        """
        adj = self.graph_discoverer(x, prev_adj)

        token_node_scores = torch.sigmoid(self.token_node_mapper(x))
        B, L, N = token_node_scores.shape
        seq_adj = torch.bmm(
            token_node_scores, torch.bmm(adj, token_node_scores.transpose(1, 2)))
        seq_adj = seq_adj / (seq_adj.max(dim=-1, keepdim=True).values + 1e-8)

        x = x + self.dropout(self.causal_attn(self.norm1(x), seq_adj[0]))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, adj


class EmbeddedModeCRT(nn.Module):
    """
    CRT embedded into Transformer layers.

    Every layer discovers/refines a causal graph from the hidden states,
    then uses that graph to modulate attention. The causal structure
    evolves from layer to layer — early layers find surface patterns,
    deeper layers discover abstract causal relationships.

    Can function as:
    - A standalone causal language model (with its own embeddings)
    - An adapter that wraps around an existing Transformer
    """

    def __init__(self, vocab_size: int, num_nodes: int,
                 d_model: int = 256, num_heads: int = 4,
                 num_layers: int = 4, d_ff: int = 512,
                 dropout: float = 0.1, causal_weight: float = 0.8,
                 hidden_dim: int = 64, max_seq_len: int = 512,
                 num_classes: int = 0):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02)

        self.layers = nn.ModuleList([
            EmbeddedCausalLayer(
                d_model, num_heads, d_ff, num_nodes,
                dropout, causal_weight, hidden_dim)
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.has_classifier = num_classes > 0
        if self.has_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
                ) -> dict[str, torch.Tensor]:
        B, L = input_ids.shape
        x = self.embedding(input_ids) + self.pos_embedding[:, :L]

        layer_adjs = []
        adj = None
        for layer in self.layers:
            x, adj = layer(x, adj)
            layer_adjs.append(adj)

        x = self.output_norm(x)

        lm_logits = self.lm_head(x)

        d = self.num_nodes
        mean_adj = adj.mean(dim=0)
        M = torch.eye(d, device=mean_adj.device) + mean_adj / d
        E = torch.matrix_power(M, d)
        dag_penalty = torch.trace(E) - d

        result = {
            "lm_logits": lm_logits,
            "hidden_states": x,
            "layer_adjs": layer_adjs,
            "final_adj": adj,
            "dag_penalty": dag_penalty,
        }

        if self.has_classifier:
            if attention_mask is not None:
                mask_f = attention_mask.unsqueeze(-1).float()
                pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)
            result["class_logits"] = self.classifier(pooled)

        return result

    def loss(self, output: dict,
             true_adj: Optional[torch.Tensor] = None,
             lm_targets: Optional[torch.Tensor] = None,
             class_targets: Optional[torch.Tensor] = None,
             weights: Optional[dict[str, float]] = None
             ) -> dict[str, torch.Tensor]:
        w = {
            "lm": 1.0, "adjacency": 0.5, "dag": 0.1,
            "sparsity": 0.05, "classification": 1.0,
            "graph_evolution": 0.05,
        }
        if weights:
            w.update(weights)

        device = output["lm_logits"].device
        total = torch.tensor(0.0, device=device)
        losses = {}

        if lm_targets is not None:
            logits = output["lm_logits"][:, :-1].contiguous()
            targets = lm_targets[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=0)
            losses["lm"] = lm_loss
            total = total + w["lm"] * lm_loss

        if true_adj is not None:
            final_adj = output["final_adj"]
            true_exp = true_adj.unsqueeze(0).expand_as(final_adj)
            adj_loss = F.binary_cross_entropy(
                final_adj.clamp(1e-6, 1 - 1e-6), true_exp)
            losses["adjacency"] = adj_loss
            total = total + w["adjacency"] * adj_loss

        dag_loss = output["dag_penalty"]
        losses["dag"] = dag_loss
        total = total + w["dag"] * dag_loss

        sparsity = output["final_adj"].mean()
        losses["sparsity"] = sparsity
        total = total + w["sparsity"] * sparsity

        layer_adjs = output["layer_adjs"]
        evolution = torch.tensor(0.0, device=device)
        if len(layer_adjs) >= 2:
            for i in range(1, len(layer_adjs)):
                evolution = evolution + (layer_adjs[i] - layer_adjs[i-1]).pow(2).mean()
            evolution = evolution / (len(layer_adjs) - 1)
        losses["graph_evolution"] = evolution
        total = total + w["graph_evolution"] * evolution

        if class_targets is not None and self.has_classifier:
            cls_loss = F.cross_entropy(output["class_logits"], class_targets)
            losses["classification"] = cls_loss
            total = total + w["classification"] * cls_loss

        losses["total"] = total
        return losses

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            output = self.forward(input_ids)
            logits = output["lm_logits"][:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_crt(mode: Literal["tool", "embedded"],
               vocab_size: int = 256,
               num_nodes: int = 8,
               **kwargs) -> nn.Module:
    """
    Factory function: create a CRT model in the specified mode.

    Args:
        mode: "tool" or "embedded"
        vocab_size: vocabulary size
        num_nodes: number of causal variables
        **kwargs: mode-specific arguments

    Tool mode kwargs:
        embed_dim, query_dim, initial_k, max_steps, hidden_dim, node_names

    Embedded mode kwargs:
        d_model, num_heads, num_layers, d_ff, dropout, causal_weight,
        hidden_dim, max_seq_len, num_classes
    """
    if mode == "tool":
        return ToolModeCRT(vocab_size=vocab_size, num_nodes=num_nodes, **kwargs)
    elif mode == "embedded":
        return EmbeddedModeCRT(vocab_size=vocab_size, num_nodes=num_nodes, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'tool' or 'embedded'.")
