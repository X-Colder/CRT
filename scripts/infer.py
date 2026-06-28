#!/usr/bin/env python
"""
Inference script for dual-mode CRT.

Tool mode:   interactive causal Q&A
Embedded mode: causal text generation / classification

Usage:
  python3 scripts/infer.py --mode tool --checkpoint checkpoints/unified/best.pt
  python3 scripts/infer.py --mode embedded --checkpoint checkpoints/unified/best.pt
  python3 scripts/infer.py --mode tool --checkpoint ... --context "..." --query "..."
"""

import argparse
import json
import torch
from pathlib import Path

from crt.integration.dual_mode import create_crt, ToolModeCRT, EmbeddedModeCRT


def parse_args():
    p = argparse.ArgumentParser(description="CRT Inference")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--mode", choices=["tool", "embedded"], default=None,
                   help="Override mode (auto-detected from checkpoint)")
    p.add_argument("--context", type=str, default=None,
                   help="Context text (non-interactive mode)")
    p.add_argument("--query", type=str, default=None,
                   help="Query text (tool mode, non-interactive)")
    p.add_argument("--prompt", type=str, default=None,
                   help="Generation prompt (embedded mode, non-interactive)")
    p.add_argument("--max-tokens", type=int, default=50)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output", type=str, default=None,
                   help="Save result to JSON file")
    return p.parse_args()


def char_tokenize(text, max_len):
    ids = [min(ord(c), 255) for c in text[:max_len]]
    pad_len = max_len - len(ids)
    mask = [1] * len(ids) + [0] * pad_len
    ids = ids + [0] * pad_len
    return (torch.tensor(ids, dtype=torch.long).unsqueeze(0),
            torch.tensor(mask, dtype=torch.long).unsqueeze(0))


def load_model(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    mode = args.mode or ckpt.get("mode", train_args.get("mode", "tool"))
    node_names = ckpt.get("node_names", None)

    if mode == "tool":
        model = create_crt("tool",
                           vocab_size=train_args["vocab_size"],
                           num_nodes=train_args["num_nodes"],
                           embed_dim=train_args["embed_dim"],
                           query_dim=train_args["query_dim"],
                           initial_k=train_args["initial_k"],
                           max_steps=train_args["max_steps"],
                           hidden_dim=train_args["hidden_dim"],
                           node_names=node_names)
    else:
        model = create_crt("embedded",
                           vocab_size=train_args["vocab_size"],
                           num_nodes=train_args["num_nodes"],
                           d_model=train_args["d_model"],
                           num_heads=train_args["num_heads"],
                           num_layers=train_args["num_layers"],
                           d_ff=train_args["d_ff"],
                           causal_weight=train_args["causal_weight"],
                           hidden_dim=train_args["hidden_dim"],
                           num_classes=train_args.get("num_classes", 0))

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded {mode} model from epoch {ckpt['epoch']}")
    return model, mode, device


def run_tool_inference(model, context, query, device, threshold=0.3):
    ctx_ids, ctx_mask = char_tokenize(context, 128)
    q_ids, q_mask = char_tokenize(query, 64)

    ctx_ids = ctx_ids.to(device)
    ctx_mask = ctx_mask.to(device)
    q_ids = q_ids.to(device)
    q_mask = q_mask.to(device)

    result = model.reason(ctx_ids, q_ids, ctx_mask, q_mask, threshold=threshold)
    return result


def run_embedded_inference(model, prompt, device, max_tokens=50, temperature=0.8):
    ids, _ = char_tokenize(prompt, len(prompt))
    ids = ids.to(device)

    with torch.no_grad():
        output = model.forward(ids)
        generated = model.generate(ids, max_new_tokens=max_tokens,
                                    temperature=temperature)

    gen_text = ""
    for c in generated[0, ids.shape[1]:].cpu().tolist():
        if 32 <= c < 127:
            gen_text += chr(c)

    layer_adjs = output["layer_adjs"]
    adj_evolution = []
    for i, adj in enumerate(layer_adjs):
        adj_evolution.append({
            "layer": i,
            "mean_edge_weight": adj.mean().item(),
            "max_edge_weight": adj.max().item(),
            "num_strong_edges": (adj > 0.5).sum().item(),
        })

    return {
        "generated_text": gen_text,
        "adj_evolution": adj_evolution,
        "dag_penalty": output["dag_penalty"].item(),
    }


def print_tool_result(result):
    print("\n" + "=" * 60)
    print("CAUSAL REASONING RESULT")
    print("=" * 60)

    print(f"\nReasoning steps: {result['reasoning_steps']}")
    conv = result["convergence"]
    print(f"Convergence: structural={conv['structural']:.3f}  "
          f"predictive={conv['predictive']:.3f}  "
          f"marginal={conv['marginal']:.3f}")

    print(f"\nActive nodes ({len(result['active_nodes'])}):")
    for node in sorted(result["active_nodes"], key=lambda x: -x["activation"]):
        bar = "#" * int(node["activation"] * 20)
        print(f"  {node['name']:15s} activation={node['activation']:.3f} {bar}")

    print(f"\nCausal edges ({len(result['causal_edges'])}):")
    for edge in sorted(result["causal_edges"], key=lambda x: -x["strength"]):
        print(f"  {edge['cause']:12s} -> {edge['effect']:12s}  "
              f"strength={edge['strength']:.3f}  "
              f"confidence={edge['chain_confidence']:.3f}")

    if result["causal_edges"]:
        chains = []
        edges = {(e["cause"], e["effect"]): e for e in result["causal_edges"]}
        causes = {e["cause"] for e in result["causal_edges"]}
        effects = {e["effect"] for e in result["causal_edges"]}
        roots = causes - effects
        for root in roots:
            chain = [root]
            current = root
            while current in {e["cause"] for e in result["causal_edges"]}:
                next_nodes = [e["effect"] for e in result["causal_edges"]
                              if e["cause"] == current]
                if next_nodes:
                    current = next_nodes[0]
                    chain.append(current)
                else:
                    break
            if len(chain) > 1:
                chains.append(chain)

        if chains:
            print("\nCausal chains:")
            for chain in chains:
                print(f"  {' -> '.join(chain)}")

    print("=" * 60)


def print_embedded_result(result):
    print("\n" + "=" * 60)
    print("CAUSAL LANGUAGE MODEL RESULT")
    print("=" * 60)

    print(f"\nGenerated: {result['generated_text']}")
    print(f"\nDAG penalty: {result['dag_penalty']:.4f}")

    print("\nCausal graph evolution across layers:")
    for info in result["adj_evolution"]:
        print(f"  Layer {info['layer']}: "
              f"mean_w={info['mean_edge_weight']:.3f}  "
              f"strong_edges={info['num_strong_edges']}")

    print("=" * 60)


def interactive_tool(model, device, threshold):
    print("\n=== CRT Tool Mode - Interactive Causal Reasoning ===")
    print("Enter context and query. Type 'quit' to exit.\n")

    while True:
        context = input("Context: ").strip()
        if context.lower() == "quit":
            break
        query = input("Query:   ").strip()
        if query.lower() == "quit":
            break

        result = run_tool_inference(model, context, query, device, threshold)
        print_tool_result(result)
        print()


def interactive_embedded(model, device, max_tokens, temperature):
    print("\n=== CRT Embedded Mode - Causal Language Generation ===")
    print("Enter a prompt. Type 'quit' to exit.\n")

    while True:
        prompt = input("Prompt: ").strip()
        if prompt.lower() == "quit":
            break

        result = run_embedded_inference(model, prompt, device, max_tokens, temperature)
        print_embedded_result(result)
        print()


def main():
    args = parse_args()
    model, mode, device = load_model(args)

    if mode == "tool" and args.context and args.query:
        result = run_tool_inference(model, args.context, args.query,
                                     device, args.threshold)
        print_tool_result(result)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)

    elif mode == "embedded" and (args.context or args.prompt):
        prompt = args.prompt or args.context
        result = run_embedded_inference(model, prompt, device,
                                        args.max_tokens, args.temperature)
        print_embedded_result(result)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)

    elif mode == "tool":
        interactive_tool(model, device, args.threshold)

    else:
        interactive_embedded(model, device, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()
