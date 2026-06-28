#!/usr/bin/env python
"""
Training script for the Adaptive Causal Model.

Supports:
- Single GPU / CPU / MPS:  python3 scripts/train.py
- Multi-GPU DDP:           torchrun --nproc_per_node=4 scripts/train.py
- Multi-node DDP:          torchrun --nnodes=2 --nproc_per_node=4 \
                               --rdzv_backend=c10d --rdzv_endpoint=HOST:PORT \
                               scripts/train.py
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

from crt.integration.adaptive_model import AdaptiveCausalModel, HierarchicalAdaptiveModel
from crt.data.synthetic import SyntheticCausalDataset
from crt.utils.metrics import structural_hamming_distance, edge_precision_recall_f1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Adaptive Causal Model")

    g = p.add_argument_group("Data generation")
    g.add_argument("--num-nodes", type=int, default=10)
    g.add_argument("--num-graphs", type=int, default=2000)
    g.add_argument("--num-obs-samples", type=int, default=200)
    g.add_argument("--num-int-samples", type=int, default=50)
    g.add_argument("--edge-prob", type=float, default=0.3)
    g.add_argument("--equation-type", choices=["linear", "nonlinear", "mixed"],
                   default="linear")
    g.add_argument("--noise-std", type=float, default=0.5)

    g = p.add_argument_group("Model architecture")
    g.add_argument("--num-iterations", type=int, default=3,
                   help="Number of graph revision iterations")
    g.add_argument("--hidden-dim", type=int, default=64)
    g.add_argument("--hierarchical", action="store_true",
                   help="Use hierarchical multi-level causal graph")
    g.add_argument("--level-sizes", type=int, nargs="+", default=None,
                   help="Node counts per level for hierarchical model "
                        "(e.g. --level-sizes 8 4 2). Overrides --num-nodes.")

    g = p.add_argument_group("Training")
    g.add_argument("--obs-batch-size", type=int, default=64)
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight-decay", type=float, default=1e-5)
    g.add_argument("--epochs", type=int, default=100)

    g = p.add_argument_group("Loss weights")
    g.add_argument("--w-pred", type=float, default=1.0)
    g.add_argument("--w-adj", type=float, default=1.0)
    g.add_argument("--w-int", type=float, default=0.5)
    g.add_argument("--w-dag", type=float, default=0.1)
    g.add_argument("--w-sparse", type=float, default=0.05)
    g.add_argument("--w-conv", type=float, default=0.1)

    g = p.add_argument_group("Output")
    g.add_argument("--log-dir", type=str, default="runs/crt")
    g.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    g.add_argument("--save-every", type=int, default=10)

    g = p.add_argument_group("Environment")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", type=str, default="auto")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed():
    if "RANK" not in os.environ:
        return
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict], obs_batch_size: int = 64) -> dict[str, torch.Tensor]:
    observations = []
    for item in batch:
        obs = item["observations"]
        n = obs.shape[0]
        if n > obs_batch_size:
            idx = torch.randperm(n)[:obs_batch_size]
            obs = obs[idx]
        observations.append(obs.mean(dim=0))

    return {
        "observations": torch.stack(observations),
        "interventions": torch.stack([item["interventions"] for item in batch]),
        "adjacency": torch.stack([item["adjacency"] for item in batch]),
        "intervention_values": torch.stack([item["intervention_values"] for item in batch]),
    }


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, weights, device, writer,
                    epoch, global_step, is_hierarchical=False):
    model.train()
    epoch_losses = {}
    count = 0
    raw_model = model.module if isinstance(model, DDP) else model

    for batch in dataloader:
        obs = batch["observations"].to(device)
        adj = batch["adjacency"].to(device)
        interventions = batch["interventions"].to(device)
        int_values = batch["intervention_values"].to(device)

        if is_hierarchical:
            output = raw_model(obs)
            losses = raw_model.loss(output, adj[0], obs, weights)
        else:
            output = raw_model(obs, interventions, int_values)
            losses = raw_model.loss(output, adj[0], obs, interventions, int_values, weights)

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        for k, v in losses.items():
            val = v.item()
            epoch_losses[k] = epoch_losses.get(k, 0.0) + val
            if is_main_process() and writer:
                writer.add_scalar(f"train_batch/{k}", val, global_step)
        count += 1
        global_step += 1

    return {k: v / max(count, 1) for k, v in epoch_losses.items()}, global_step


@torch.no_grad()
def validate(model, dataloader, weights, device, is_hierarchical=False):
    model.eval()
    raw_model = model.module if isinstance(model, DDP) else model
    epoch_losses = {}
    all_shd = []
    all_f1 = []
    count = 0

    for batch in dataloader:
        obs = batch["observations"].to(device)
        adj = batch["adjacency"].to(device)
        interventions = batch["interventions"].to(device)
        int_values = batch["intervention_values"].to(device)

        if is_hierarchical:
            output = raw_model(obs)
            losses = raw_model.loss(output, adj[0], obs, weights)
        else:
            output = raw_model(obs, interventions, int_values)
            losses = raw_model.loss(output, adj[0], obs, interventions, int_values, weights)

        for k, v in losses.items():
            epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
        count += 1

        final_adj = output["final_adj"].mean(0).cpu()
        true_adj = adj[0].cpu()
        all_shd.append(structural_hamming_distance(final_adj, true_adj))
        prf = edge_precision_recall_f1(final_adj, true_adj)
        all_f1.append(prf["f1"])

    avg_losses = {k: v / max(count, 1) for k, v in epoch_losses.items()}
    avg_losses["shd"] = np.mean(all_shd)
    avg_losses["f1"] = np.mean(all_f1)
    return avg_losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    setup_distributed()

    torch.manual_seed(args.seed + get_rank())
    np.random.seed(args.seed + get_rank())

    if is_distributed():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if is_main_process():
        print(f"Using device: {device}")
        if is_distributed():
            print(f"Distributed: {get_world_size()} processes")

    # Determine model type and effective num_nodes
    is_hierarchical = args.hierarchical
    if is_hierarchical and args.level_sizes:
        level_sizes = args.level_sizes
        effective_nodes = sum(level_sizes)
    elif is_hierarchical:
        n = args.num_nodes
        level_sizes = [n, max(n // 2, 2), max(n // 4, 1)]
        effective_nodes = sum(level_sizes)
    else:
        effective_nodes = args.num_nodes

    if is_main_process():
        print("Generating synthetic dataset...")
    dataset = SyntheticCausalDataset(
        num_graphs=args.num_graphs, num_nodes=effective_nodes,
        edge_prob=args.edge_prob, num_obs_samples=args.num_obs_samples,
        num_int_samples=args.num_int_samples, equation_type=args.equation_type,
        noise_std=args.noise_std, seed=args.seed)

    n_train = int(len(dataset) * 0.8)
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_set = Subset(dataset, indices[:n_train])
    val_set = Subset(dataset, indices[n_train:])

    collate = partial(collate_fn, obs_batch_size=args.obs_batch_size)

    train_sampler = DistributedSampler(train_set) if is_distributed() else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if is_distributed() else None

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        collate_fn=collate)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, sampler=val_sampler,
        collate_fn=collate)

    if is_hierarchical:
        model = HierarchicalAdaptiveModel(
            level_sizes=level_sizes, num_iterations=args.num_iterations,
            hidden_dim=args.hidden_dim).to(device)
    else:
        model = AdaptiveCausalModel(
            num_nodes=args.num_nodes, num_iterations=args.num_iterations,
            hidden_dim=args.hidden_dim).to(device)

    if is_distributed():
        model = DDP(model, device_ids=[device.index])

    if is_main_process():
        raw = model.module if isinstance(model, DDP) else model
        print(f"Model: {'Hierarchical' if is_hierarchical else 'Flat'} "
              f"({sum(p.numel() for p in raw.parameters()):,} parameters)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    weights = {
        "prediction": args.w_pred, "adjacency": args.w_adj,
        "intervention": args.w_int, "dag": args.w_dag,
        "sparsity": args.w_sparse, "convergence": args.w_conv,
    }

    ckpt_dir = Path(args.checkpoint_dir)
    writer = None
    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(args.log_dir)

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_losses, global_step = train_one_epoch(
            model, train_loader, optimizer, weights, device, writer, epoch,
            global_step, is_hierarchical)
        val_losses = validate(model, val_loader, weights, device, is_hierarchical)
        scheduler.step()

        if is_main_process():
            for k, v in train_losses.items():
                writer.add_scalar(f"train_epoch/{k}", v, epoch)
            for k, v in val_losses.items():
                writer.add_scalar(f"val_epoch/{k}", v, epoch)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

            print(f"Epoch {epoch:03d} | "
                  f"train_loss={train_losses['total']:.4f} | "
                  f"val_loss={val_losses['total']:.4f} | "
                  f"val_shd={val_losses['shd']:.1f} | "
                  f"val_f1={val_losses['f1']:.3f}")

            raw_model = model.module if isinstance(model, DDP) else model

            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                save_args = vars(args).copy()
                if is_hierarchical:
                    save_args["level_sizes"] = level_sizes
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_losses["total"],
                    "args": save_args,
                }, ckpt_dir / "best.pt")

            if epoch % args.save_every == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "args": save_args if is_hierarchical else vars(args),
                }, ckpt_dir / f"epoch_{epoch:03d}.pt")

    if is_main_process():
        raw_model = model.module if isinstance(model, DDP) else model
        torch.save({
            "epoch": args.epochs,
            "model_state_dict": raw_model.state_dict(),
            "args": vars(args),
        }, ckpt_dir / "final.pt")

        writer.close()
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to {ckpt_dir}")
        print(f"TensorBoard logs at {args.log_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
