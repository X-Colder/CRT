"""
Synthetic SCM data generation for causal structure learning.

Generates random DAGs with known ground-truth adjacency matrices,
samples observational and interventional data from structural causal models.
"""

import torch
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
from typing import Optional, Literal


def random_dag(num_nodes: int, edge_prob: float = 0.3,
               rng: Optional[np.random.Generator] = None) -> nx.DiGraph:
    rng = rng or np.random.default_rng()
    perm = rng.permutation(num_nodes)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() < edge_prob:
                adj[perm[i], perm[j]] = 1.0
    g = nx.DiGraph()
    g.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i, j] > 0:
                g.add_edge(i, j)
    return g


def dag_to_adjacency(dag: nx.DiGraph, num_nodes: int) -> torch.Tensor:
    adj = torch.zeros(num_nodes, num_nodes)
    for u, v in dag.edges():
        adj[u, v] = 1.0
    return adj


class StructuralCausalModel:
    def __init__(self, adjacency: torch.Tensor,
                 equation_type: Literal["linear", "nonlinear", "mixed"] = "linear",
                 noise_std: float = 0.5,
                 rng: Optional[np.random.Generator] = None):
        self.adjacency = adjacency.numpy() if isinstance(adjacency, torch.Tensor) else adjacency
        self.num_nodes = self.adjacency.shape[0]
        self.equation_type = equation_type
        self.noise_std = noise_std
        self.rng = rng or np.random.default_rng()

        g = nx.DiGraph()
        g.add_nodes_from(range(self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.adjacency[i, j] > 0:
                    g.add_edge(i, j)
        self.topo_order = list(nx.topological_sort(g))

        self.weights = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for j in range(self.num_nodes):
            for i in range(self.num_nodes):
                if self.adjacency[i, j] > 0:
                    sign = self.rng.choice([-1, 1])
                    mag = self.rng.uniform(0.5, 2.0)
                    self.weights[i, j] = sign * mag

        if equation_type == "mixed":
            self.node_nonlinear = self.rng.choice([False, True], size=self.num_nodes)
        elif equation_type == "nonlinear":
            self.node_nonlinear = np.ones(self.num_nodes, dtype=bool)
        else:
            self.node_nonlinear = np.zeros(self.num_nodes, dtype=bool)

    def sample_observational(self, num_samples: int) -> torch.Tensor:
        data = np.zeros((num_samples, self.num_nodes), dtype=np.float32)
        noise = self.rng.normal(0, self.noise_std, size=(num_samples, self.num_nodes)).astype(np.float32)
        for j in self.topo_order:
            parent_mask = self.adjacency[:, j] > 0
            if not parent_mask.any():
                data[:, j] = noise[:, j]
            else:
                linear_comb = data @ self.weights[:, j]
                if self.node_nonlinear[j]:
                    data[:, j] = np.tanh(linear_comb) + noise[:, j]
                else:
                    data[:, j] = linear_comb + noise[:, j]
        return torch.from_numpy(data)

    def sample_interventional(self, num_samples: int,
                              intervention_node: int,
                              intervention_value: float) -> torch.Tensor:
        data = np.zeros((num_samples, self.num_nodes), dtype=np.float32)
        noise = self.rng.normal(0, self.noise_std, size=(num_samples, self.num_nodes)).astype(np.float32)
        for j in self.topo_order:
            if j == intervention_node:
                data[:, j] = intervention_value
                continue
            parent_mask = self.adjacency[:, j] > 0
            if not parent_mask.any():
                data[:, j] = noise[:, j]
            else:
                linear_comb = data @ self.weights[:, j]
                if self.node_nonlinear[j]:
                    data[:, j] = np.tanh(linear_comb) + noise[:, j]
                else:
                    data[:, j] = linear_comb + noise[:, j]
        return torch.from_numpy(data)


class SyntheticCausalDataset(Dataset):
    def __init__(self, num_graphs: int = 1000, num_nodes: int = 10,
                 edge_prob: float = 0.3, num_obs_samples: int = 200,
                 num_int_samples: int = 50,
                 equation_type: Literal["linear", "nonlinear", "mixed"] = "linear",
                 noise_std: float = 0.5, seed: int = 42):
        self.num_nodes = num_nodes
        self.data = []
        rng = np.random.default_rng(seed)

        for _ in range(num_graphs):
            dag = random_dag(num_nodes, edge_prob, rng)
            adj = dag_to_adjacency(dag, num_nodes)
            scm = StructuralCausalModel(adj, equation_type, noise_std,
                                        np.random.default_rng(rng.integers(0, 2**31)))

            obs = scm.sample_observational(num_obs_samples)

            int_data = []
            int_values = []
            obs_np = obs.numpy()
            for node_idx in range(num_nodes):
                val = float(rng.uniform(obs_np[:, node_idx].min() - 1,
                                        obs_np[:, node_idx].max() + 1))
                int_values.append(val)
                int_data.append(scm.sample_interventional(num_int_samples, node_idx, val))

            self.data.append({
                "observations": obs,
                "interventions": torch.stack(int_data),
                "adjacency": adj,
                "intervention_values": torch.tensor(int_values),
            })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.data[idx]
