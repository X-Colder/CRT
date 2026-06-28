"""CRT - Causal Reasoning Transformer."""

from crt.graph.causal_graph import CausalGraph, DifferentiableCausalGraph
from crt.graph.hierarchical_graph import HierarchicalCausalGraph
from crt.transformer.causal_attention import CausalConstrainedAttention
from crt.integration.perception_reasoning import PerceptionReasoningModel
from crt.integration.hypothesis_verifier import HypothesisVerifier
from crt.integration.adaptive_model import AdaptiveCausalModel, HierarchicalAdaptiveModel
from crt.integration.sparse_reasoning import SparseReasoningEngine
from crt.integration.semantic_reasoning import SemanticCausalModel
from crt.integration.dual_mode import ToolModeCRT, EmbeddedModeCRT, create_crt

__version__ = "0.1.0"

__all__ = [
    "CausalGraph",
    "DifferentiableCausalGraph",
    "HierarchicalCausalGraph",
    "CausalConstrainedAttention",
    "PerceptionReasoningModel",
    "HypothesisVerifier",
    "AdaptiveCausalModel",
    "HierarchicalAdaptiveModel",
    "SparseReasoningEngine",
    "SemanticCausalModel",
    "ToolModeCRT",
    "EmbeddedModeCRT",
    "create_crt",
]
