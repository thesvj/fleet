#!/usr/bin/env python3
"""Tiny multi-rank training loop using fleet helpers."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("torch required", file=sys.stderr)
        return 2

    import fleet

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-2)
    args = p.parse_args()

    fleet.init()
    rank, world = fleet.rank(), fleet.world_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    for step in range(args.steps):
        if fleet.checkpoint_if_needed():
            if rank == 0:
                print(f"[toy] checkpoint at step {step}", flush=True)
            break
        x = torch.randn(64, 32, device=device)
        y = torch.randn(64, 1, device=device)
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        fleet.step(opt, model)
        if step % 5 == 0 or step == args.steps - 1:
            print(f"[toy] rank={rank}/{world} step={step} loss={loss.item():.4f}", flush=True)

    fleet.barrier()
    if rank == 0:
        print("[toy] done", flush=True)
    fleet.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
