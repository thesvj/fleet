#!/usr/bin/env python3
"""Torch allreduce smoke across the pool (gloo or nccl)."""

from __future__ import annotations

import os
import sys

import fleet


def main() -> int:
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        print("torch required", file=sys.stderr)
        return 2

    info = fleet.init()
    rank, world = fleet.rank(), fleet.world_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    t = torch.ones(1, device=device) * (rank + 1)
    if world > 1:
        if not dist.is_initialized():
            print("process group not initialized", file=sys.stderr)
            return 2
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        expected = world * (world + 1) / 2
        if abs(t.item() - expected) > 1e-3:
            print(f"allreduce fail got={t.item()} expected={expected}", file=sys.stderr)
            return 2
        print(f"[smoke_ddp] rank={rank} sum={t.item()} ok", flush=True)
    else:
        print(f"[smoke_ddp] world=1 skip allreduce backend={info['backend']}", flush=True)

    fleet.barrier()
    if rank == 0:
        print(f"[smoke_ddp] PASS world={world} backend={info['backend']}", flush=True)
    fleet.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
