#!/usr/bin/env python
"""
Unified training script for dual-mode CRT.

Two modes:
  python3 scripts/train_unified.py --mode tool     # CRT as external tool
  python3 scripts/train_unified.py --mode embedded  # CRT inside Transformer

Distributed:
  torchrun --nproc_per_node=4 scripts/train_unified.py --mode tool
"""

import argparse
import os
import torch
import torch.distributed as dist
import numpy as np
from functools import partial
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from crt.integration.dual_mode import create_crt
from crt.integration.semantic_reasoning import (
    CausalTextDataset, build_demo_medical_dataset)
from crt.utils.metrics import structural_hamming_distance, edge_precision_recall_f1


def parse_args():
    p = argparse.ArgumentParser(description="Train dual-mode CRT")

    p.add_argument("--mode", choices=["tool", "embedded"], required=True)

    g = p.add_argument_group("Data")
    g.add_argument("--dataset", choices=["medical_demo", "custom"],
                   default="medical_demo")
    g.add_argument("--data-path", type=str, default=None,
                   help="Path to custom dataset (JSON)")
    g.add_argument("--num-repeats", type=int, default=200,
                   help="Repeat small datasets to create enough training data")

    g = p.add_argument_group("Model (shared)")
    g.add_argument("--vocab-size", type=int, default=256)
    g.add_argument("--num-nodes", type=int, default=8)
    g.add_argument("--hidden-dim", type=int, default=64)

    g = p.add_argument_group("Model (tool mode)")
    g.add_argument("--embed-dim", type=int, default=128)
    g.add_argument("--query-dim", type=int, default=64)
    g.add_argument("--initial-k", type=int, default=4)
    g.add_argument("--max-steps", type=int, default=4)

    g = p.add_argument_group("Model (embedded mode)")
    g.add_argument("--d-model", type=int, default=128)
    g.add_argument("--num-heads", type=int, default=4)
    g.add_argument("--num-layers", type=int, default=4)
    g.add_argument("--d-ff", type=int, default=256)
    g.add_argument("--causal-weight", type=float, default=0.8)
    g.add_argument("--num-classes", type=int, default=0,
                   help="If >0, add a classification head (embedded mode)")

    g = p.add_argument_group("Training")
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--weight-decay", type=float, default=1e-5)
    g.add_argument("--epochs", type=int, default=50)
    g.add_argument("--grad-clip", type=float, default=5.0)

    g = p.add_argument_group("Output")
    g.add_argument("--log-dir", type=str, default="runs/crt_unified")
    g.add_argument("--checkpoint-dir", type=str, default="checkpoints/unified")
    g.add_argument("--save-every", type=int, default=10)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", type=str, default="auto")

    return p.parse_args()


def is_distributed():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_distributed() else 0

def is_main():
    return get_rank() == 0

def setup_distributed():
    if "RANK" not in os.environ:
        return
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def build_dataset(args):
    if args.dataset == "medical_demo":
        base_ds = build_demo_medical_dataset()
        scenarios = base_ds.scenarios * args.num_repeats
        ds = CausalTextDataset(
            node_names=base_ds.node_names,
            adjacency=base_ds.adjacency,
            scenarios=scenarios)
        return ds, base_ds.node_names, base_ds.adjacency
    else:
        raise NotImplementedError(f"Custom dataset loading not yet implemented")


def collate_tool(batch):
    return {
        "context_ids": torch.stack([b["context_ids"] for b in batch]),
        "query_ids": torch.stack([b["query_ids"] for b in batch]),
        "context_mask": torch.stack([b["context_mask"] for b in batch]),
        "query_mask": torch.stack([b["query_mask"] for b in batch]),
        "adjacency": torch.stack([b["adjacency"] for b in batch]),
        "target_nodes": torch.stack([b["target_nodes"] for b in batch]),
    }


def collate_embedded(batch):
    return {
        "context_ids": torch.stack([b["context_ids"] for b in batch]),
        "context_mask": torch.stack([b["context_mask"] for b in batch]),
        "adjacency": torch.stack([b["adjacency"] for b in batch]),
        "target_nodes": torch.stack([b["target_nodes"] for b in batch]),
    }


def train_tool_epoch(model, loader, optimizer, device, writer, epoch, global_step):
    model.train()
    raw = model.module if isinstance(model, DDP) else model
    losses_sum = {}
    count = 0

    for batch in loader:
        ctx = batch["context_ids"].to(device)
        q = batch["query_ids"].to(device)
        ctx_m = batch["context_mask"].to(device)
        q_m = batch["query_mask"].to(device)
        adj = batch["adjacency"][0].to(device)
        targets = batch["target_nodes"].to(device)

        output = raw(ctx, q, ctx_m, q_m)
        losses = raw.loss(output, adj, targets)

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        for k, v in losses.items():
            val = v.item()
            losses_sum[k] = losses_sum.get(k, 0) + val
            if is_main() and writer:
                writer.add_scalar(f"train_batch/{k}", val, global_step)
        count += 1
        global_step += 1

    return {k: v / max(count, 1) for k, v in losses_sum.items()}, global_step


def train_embedded_epoch(model, loader, optimizer, device, writer, epoch, global_step):
    model.train()
    raw = model.module if isinstance(model, DDP) else model
    losses_sum = {}
    count = 0

    for batch in loader:
        ids = batch["context_ids"].to(device)
        mask = batch["context_mask"].to(device)
        adj = batch["adjacency"][0].to(device)

        output = raw(ids, mask)
        losses = raw.loss(output, true_adj=adj, lm_targets=ids)

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        for k, v in losses.items():
            val = v.item()
            losses_sum[k] = losses_sum.get(k, 0) + val
            if is_main() and writer:
                writer.add_scalar(f"train_batch/{k}", val, global_step)
        count += 1
        global_step += 1

    return {k: v / max(count, 1) for k, v in losses_sum.items()}, global_step


@torch.no_grad()
def validate_tool(model, loader, device):
    model.eval()
    raw = model.module if isinstance(model, DDP) else model
    losses_sum = {}
    all_shd, all_f1 = [], []
    count = 0

    for batch in loader:
        ctx = batch["context_ids"].to(device)
        q = batch["query_ids"].to(device)
        ctx_m = batch["context_mask"].to(device)
        q_m = batch["query_mask"].to(device)
        adj = batch["adjacency"][0].to(device)
        targets = batch["target_nodes"].to(device)

        output = raw(ctx, q, ctx_m, q_m)
        losses = raw.loss(output, adj, targets)

        for k, v in losses.items():
            losses_sum[k] = losses_sum.get(k, 0) + v.item()

        pred_adj = output["final_adj"].mean(0).cpu()
        all_shd.append(structural_hamming_distance(pred_adj, batch["adjacency"][0]))
        all_f1.append(edge_precision_recall_f1(pred_adj, batch["adjacency"][0])["f1"])
        count += 1

    avg = {k: v / max(count, 1) for k, v in losses_sum.items()}
    avg["shd"] = np.mean(all_shd)
    avg["f1"] = np.mean(all_f1)
    return avg


@torch.no_grad()
def validate_embedded(model, loader, device):
    model.eval()
    raw = model.module if isinstance(model, DDP) else model
    losses_sum = {}
    all_shd, all_f1 = [], []
    count = 0

    for batch in loader:
        ids = batch["context_ids"].to(device)
        mask = batch["context_mask"].to(device)
        adj = batch["adjacency"][0].to(device)

        output = raw(ids, mask)
        losses = raw.loss(output, true_adj=adj, lm_targets=ids)

        for k, v in losses.items():
            losses_sum[k] = losses_sum.get(k, 0) + v.item()

        pred_adj = output["final_adj"].mean(0).cpu()
        all_shd.append(structural_hamming_distance(pred_adj, batch["adjacency"][0]))
        all_f1.append(edge_precision_recall_f1(pred_adj, batch["adjacency"][0])["f1"])
        count += 1

    avg = {k: v / max(count, 1) for k, v in losses_sum.items()}
    avg["shd"] = np.mean(all_shd)
    avg["f1"] = np.mean(all_f1)
    return avg


def main():
    args = parse_args()
    setup_distributed()
    torch.manual_seed(args.seed + get_rank())
    np.random.seed(args.seed + get_rank())

    if is_distributed():
        device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if is_main():
        print(f"Mode: {args.mode}")
        print(f"Device: {device}")
        if is_distributed():
            print(f"World size: {dist.get_world_size()}")

    dataset, node_names, true_adj = build_dataset(args)
    n_train = int(len(dataset) * 0.8)
    indices = torch.randperm(len(dataset),
                             generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_set = Subset(dataset, indices[:n_train])
    val_set = Subset(dataset, indices[n_train:])

    collate = collate_tool if args.mode == "tool" else collate_embedded
    train_sampler = DistributedSampler(train_set) if is_distributed() else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if is_distributed() else None
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, sampler=val_sampler, collate_fn=collate)

    if args.mode == "tool":
        model = create_crt("tool", vocab_size=args.vocab_size,
                           num_nodes=args.num_nodes, embed_dim=args.embed_dim,
                           query_dim=args.query_dim, initial_k=args.initial_k,
                           max_steps=args.max_steps, hidden_dim=args.hidden_dim,
                           node_names=node_names).to(device)
    else:
        model = create_crt("embedded", vocab_size=args.vocab_size,
                           num_nodes=args.num_nodes, d_model=args.d_model,
                           num_heads=args.num_heads, num_layers=args.num_layers,
                           d_ff=args.d_ff, causal_weight=args.causal_weight,
                           hidden_dim=args.hidden_dim,
                           num_classes=args.num_classes).to(device)

    if is_distributed():
        model = DDP(model, device_ids=[device.index])

    if is_main():
        raw = model.module if isinstance(model, DDP) else model
        print(f"Parameters: {sum(p.numel() for p in raw.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = Path(args.checkpoint_dir)
    writer = None
    if is_main():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(args.log_dir)

    train_fn = train_tool_epoch if args.mode == "tool" else train_embedded_epoch
    val_fn = validate_tool if args.mode == "tool" else validate_embedded

    best_val = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        train_losses, global_step = train_fn(
            model, train_loader, optimizer, device, writer, epoch, global_step)
        val_losses = val_fn(model, val_loader, device)
        scheduler.step()

        if is_main():
            for k, v in train_losses.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_losses.items():
                writer.add_scalar(f"val/{k}", v, epoch)

            print(f"Epoch {epoch:03d} | "
                  f"train={train_losses['total']:.4f} | "
                  f"val={val_losses['total']:.4f} | "
                  f"shd={val_losses['shd']:.1f} | "
                  f"f1={val_losses['f1']:.3f}")

            raw = model.module if isinstance(model, DDP) else model
            save_dict = {
                "epoch": epoch,
                "model_state_dict": raw.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "mode": args.mode,
                "node_names": node_names,
            }

            if val_losses["total"] < best_val:
                best_val = val_losses["total"]
                save_dict["val_loss"] = best_val
                torch.save(save_dict, ckpt_dir / "best.pt")

            if epoch % args.save_every == 0:
                torch.save(save_dict, ckpt_dir / f"epoch_{epoch:03d}.pt")

    if is_main():
        writer.close()
        print(f"Done. Best val loss: {best_val:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
