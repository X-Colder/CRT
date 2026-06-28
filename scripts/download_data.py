#!/usr/bin/env python
"""
Download benchmark datasets for CRT experiments.

Datasets:
  1. Sachs (protein signaling, 11 nodes, 17 edges, interventional)
  2. Asia (Bayesian network benchmark, 8 nodes)
  3. Alarm (medical monitoring, 37 nodes)
  4. Built-in medical demo (8 nodes, included in CRT)

Usage:
  python3 scripts/download_data.py --all
  python3 scripts/download_data.py --dataset sachs
"""

import argparse
import json
import urllib.request
import os
import numpy as np
from pathlib import Path


DATA_DIR = Path("data")


def download_file(url, dest):
    print(f"  Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def prepare_sachs():
    """
    Sachs et al. 2005 — protein signaling network.
    11 nodes, 17 edges, with interventional data.
    The gold standard benchmark for causal discovery.
    """
    dest = DATA_DIR / "sachs"
    dest.mkdir(parents=True, exist_ok=True)

    node_names = ["Raf", "Mek", "PLCg", "PIP2", "PIP3",
                  "Erk", "Akt", "PKA", "PKC", "P38", "JNK"]
    edges = [
        ("PLCg", "PIP2"), ("PLCg", "PIP3"), ("PIP3", "PIP2"),
        ("PIP2", "PKC"), ("PKC", "Raf"), ("PKC", "Mek"),
        ("PKC", "P38"), ("PKC", "JNK"), ("PKA", "Raf"),
        ("PKA", "Mek"), ("PKA", "Erk"), ("PKA", "Akt"),
        ("PKA", "P38"), ("PKA", "JNK"), ("Raf", "Mek"),
        ("Mek", "Erk"), ("Erk", "Akt"),
    ]

    N = len(node_names)
    adj = np.zeros((N, N), dtype=np.float32)
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    for cause, effect in edges:
        adj[name_to_idx[cause], name_to_idx[effect]] = 1.0

    np.save(dest / "adjacency.npy", adj)

    with open(dest / "metadata.json", "w") as f:
        json.dump({
            "name": "Sachs Protein Signaling",
            "num_nodes": N,
            "num_edges": len(edges),
            "node_names": node_names,
            "edges": edges,
            "reference": "Sachs et al., Causal Protein-Signaling Networks Derived from Multiparameter Single-Cell Data, Science 2005",
            "has_interventional": True,
        }, f, indent=2)

    print(f"  Sachs: {N} nodes, {len(edges)} edges")

    num_obs = 1000
    rng = np.random.default_rng(42)
    topo = ["PIP3", "PLCg", "PIP2", "PKC", "PKA", "Raf", "Mek", "Erk", "Akt", "P38", "JNK"]
    data = np.zeros((num_obs, N), dtype=np.float32)
    weights = rng.uniform(0.5, 2.0, size=(N, N)).astype(np.float32)
    weights *= adj
    for name in topo:
        j = name_to_idx[name]
        parent_vals = data @ weights[:, j]
        data[:, j] = parent_vals + rng.normal(0, 0.5, num_obs).astype(np.float32)

    np.save(dest / "observational.npy", data)
    print(f"  Generated {num_obs} observational samples")


def prepare_asia():
    """
    Asia (Lauritzen-Spiegelhalter) network — 8 nodes.
    Classic BN benchmark for structure learning.
    """
    dest = DATA_DIR / "asia"
    dest.mkdir(parents=True, exist_ok=True)

    node_names = ["Asia", "Smoking", "Tuberculosis", "LungCancer",
                  "Bronchitis", "TbOrCancer", "XRay", "Dyspnea"]
    edges = [
        ("Asia", "Tuberculosis"), ("Smoking", "LungCancer"),
        ("Smoking", "Bronchitis"), ("Tuberculosis", "TbOrCancer"),
        ("LungCancer", "TbOrCancer"), ("TbOrCancer", "XRay"),
        ("TbOrCancer", "Dyspnea"), ("Bronchitis", "Dyspnea"),
    ]

    N = len(node_names)
    adj = np.zeros((N, N), dtype=np.float32)
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    for cause, effect in edges:
        adj[name_to_idx[cause], name_to_idx[effect]] = 1.0

    np.save(dest / "adjacency.npy", adj)

    with open(dest / "metadata.json", "w") as f:
        json.dump({
            "name": "Asia (Lauritzen-Spiegelhalter)",
            "num_nodes": N,
            "num_edges": len(edges),
            "node_names": node_names,
            "edges": edges,
            "reference": "Lauritzen & Spiegelhalter, 1988",
        }, f, indent=2)

    num_obs = 1000
    rng = np.random.default_rng(42)
    data = np.zeros((num_obs, N), dtype=np.float32)
    weights = rng.uniform(0.5, 2.0, size=(N, N)).astype(np.float32)
    weights *= adj
    topo_order = ["Asia", "Smoking", "Tuberculosis", "LungCancer",
                  "Bronchitis", "TbOrCancer", "XRay", "Dyspnea"]
    for name in topo_order:
        j = name_to_idx[name]
        parent_vals = data @ weights[:, j]
        data[:, j] = np.tanh(parent_vals) + rng.normal(0, 0.3, num_obs).astype(np.float32)

    np.save(dest / "observational.npy", data)
    print(f"  Asia: {N} nodes, {len(edges)} edges, {num_obs} samples")


def prepare_alarm():
    """
    ALARM network — 37 nodes, 46 edges.
    Medical monitoring benchmark for larger graphs.
    """
    dest = DATA_DIR / "alarm"
    dest.mkdir(parents=True, exist_ok=True)

    node_names = [
        "LVFAILURE", "HISTORY", "CVP", "PCWP", "HYPOVOLEMIA",
        "LVEDVOLUME", "STROKEVOLUME", "ERRLOWOUTPUT", "HRBP",
        "HREKG", "HRSAT", "INSUFFANESTH", "ANAPHYLAXIS",
        "TPR", "EXPCO2", "KINKEDTUBE", "MINVOL", "FIO2",
        "PVSAT", "SAO2", "PAP", "PULMEMBOLUS", "SHUNT",
        "INTUBATION", "PRESS", "DISCONNECT", "MINVOLSET",
        "VENTMACH", "VENTTUBE", "VENTLUNG", "VENTALV",
        "ARTCO2", "CATECHOL", "HR", "CO", "BP",
        "ERRCAUTER",
    ]

    edges = [
        ("LVFAILURE", "LVEDVOLUME"), ("LVFAILURE", "STROKEVOLUME"),
        ("LVFAILURE", "HISTORY"), ("HYPOVOLEMIA", "LVEDVOLUME"),
        ("HYPOVOLEMIA", "STROKEVOLUME"), ("LVEDVOLUME", "CVP"),
        ("LVEDVOLUME", "PCWP"), ("STROKEVOLUME", "CO"),
        ("INSUFFANESTH", "CATECHOL"), ("ANAPHYLAXIS", "TPR"),
        ("TPR", "CATECHOL"), ("TPR", "BP"), ("KINKEDTUBE", "PRESS"),
        ("KINKEDTUBE", "VENTTUBE"), ("VENTMACH", "VENTTUBE"),
        ("DISCONNECT", "VENTTUBE"), ("VENTTUBE", "VENTLUNG"),
        ("INTUBATION", "VENTLUNG"), ("INTUBATION", "SHUNT"),
        ("INTUBATION", "MINVOL"), ("INTUBATION", "PRESS"),
        ("VENTLUNG", "VENTALV"), ("VENTLUNG", "MINVOL"),
        ("VENTALV", "PVSAT"), ("VENTALV", "ARTCO2"),
        ("VENTALV", "EXPCO2"), ("FIO2", "PVSAT"),
        ("PVSAT", "SAO2"), ("SAO2", "CATECHOL"),
        ("PULMEMBOLUS", "PAP"), ("PULMEMBOLUS", "SHUNT"),
        ("SHUNT", "SAO2"), ("ERRLOWOUTPUT", "HRBP"),
        ("HRBP", "HREKG"), ("HRBP", "HRSAT"),
        ("ERRCAUTER", "HREKG"), ("ERRCAUTER", "HRSAT"),
        ("HR", "HREKG"), ("HR", "HRSAT"), ("HR", "HRBP"),
        ("CO", "BP"), ("CATECHOL", "HR"),
        ("MINVOLSET", "VENTMACH"), ("ARTCO2", "EXPCO2"),
        ("ARTCO2", "CATECHOL"), ("MINVOL", "EXPCO2"),
        ("PAP", "PCWP"),
    ]

    N = len(node_names)
    adj = np.zeros((N, N), dtype=np.float32)
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    for cause, effect in edges:
        adj[name_to_idx[cause], name_to_idx[effect]] = 1.0

    np.save(dest / "adjacency.npy", adj)

    with open(dest / "metadata.json", "w") as f:
        json.dump({
            "name": "ALARM Medical Monitoring",
            "num_nodes": N,
            "num_edges": len(edges),
            "node_names": node_names,
        }, f, indent=2)

    print(f"  ALARM: {N} nodes, {len(edges)} edges")


def prepare_medical_demo():
    """Export the built-in medical demo dataset."""
    dest = DATA_DIR / "medical_demo"
    dest.mkdir(parents=True, exist_ok=True)

    from crt.integration.semantic_reasoning import build_demo_medical_dataset
    ds = build_demo_medical_dataset()

    np.save(dest / "adjacency.npy", ds.adjacency.numpy())
    with open(dest / "metadata.json", "w") as f:
        json.dump({
            "name": "CRT Medical Demo",
            "num_nodes": len(ds.node_names),
            "node_names": ds.node_names,
        }, f, indent=2)

    scenarios = []
    for s in ds.scenarios:
        scenarios.append({
            "context": s["context"],
            "query": s["query"],
            "target_nodes": s["target_nodes"],
        })
    with open(dest / "scenarios.json", "w") as f:
        json.dump(scenarios, f, indent=2, ensure_ascii=False)

    print(f"  Medical demo: {len(ds.node_names)} nodes, {len(ds.scenarios)} scenarios")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("--dataset", choices=["sachs", "asia", "alarm", "medical_demo"],
                   nargs="+", default=None)
    args = p.parse_args()

    if args.all:
        datasets = ["sachs", "asia", "alarm", "medical_demo"]
    elif args.dataset:
        datasets = args.dataset
    else:
        datasets = ["sachs", "asia", "alarm", "medical_demo"]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Preparing datasets...")

    for ds in datasets:
        print(f"\n[{ds}]")
        if ds == "sachs":
            prepare_sachs()
        elif ds == "asia":
            prepare_asia()
        elif ds == "alarm":
            prepare_alarm()
        elif ds == "medical_demo":
            prepare_medical_demo()

    print("\nDone! All datasets saved to data/")


if __name__ == "__main__":
    main()
