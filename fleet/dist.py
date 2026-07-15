"""Optional helpers for training scripts: import fleet; fleet.init()."""

from __future__ import annotations

import os
import signal
from typing import Any

_RANK = 0
_WORLD = 1
_BACKEND: str | None = None
_CKPT = False


def _signals() -> None:
    def handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        global _CKPT
        _CKPT = True
        print(f"[fleet] signal {signum} → checkpoint flag", flush=True)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handler)
        except Exception:
            pass
    if hasattr(signal, "SIGUSR1"):
        try:
            signal.signal(signal.SIGUSR1, handler)
        except Exception:
            pass


def init(backend: str | None = None, timeout_s: int = 1800) -> dict[str, Any]:
    """Init process group from RANK/WORLD_SIZE/MASTER_* set by the pool worker."""
    global _RANK, _WORLD, _BACKEND
    _RANK = int(os.environ.get("RANK", "0"))
    _WORLD = int(os.environ.get("WORLD_SIZE", "1"))
    _signals()

    if _WORLD == 1:
        _BACKEND = "none"
        return {"rank": 0, "world_size": 1, "backend": "none"}

    import datetime

    import torch
    import torch.distributed as dist

    want = (backend or os.environ.get("FLEET_BACKEND") or "auto").lower()
    if want == "auto":
        want = "nccl" if torch.cuda.is_available() else "gloo"
    if want == "nccl" and not torch.cuda.is_available():
        want = "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if dist.is_initialized():
        _BACKEND = want
        return {"rank": _RANK, "world_size": _WORLD, "backend": want}

    dist.init_process_group(
        backend=want,
        rank=_RANK,
        world_size=_WORLD,
        timeout=datetime.timedelta(seconds=timeout_s),
    )
    _BACKEND = want
    print(f"[fleet] init backend={want} rank={_RANK}/{_WORLD}", flush=True)
    return {"rank": _RANK, "world_size": _WORLD, "backend": want}


def rank() -> int:
    return _RANK


def world_size() -> int:
    return _WORLD


def is_initialized() -> bool:
    try:
        import torch.distributed as dist

        return dist.is_available() and dist.is_initialized()
    except Exception:
        return _WORLD == 1


def allreduce_grads(model: Any) -> None:
    if _WORLD <= 1:
        return
    import torch.distributed as dist

    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(_WORLD)


def step(optimizer: Any, model: Any | None = None) -> None:
    if model is not None:
        allreduce_grads(model)
    optimizer.step()


def checkpoint_if_needed() -> bool:
    return _CKPT


def barrier() -> None:
    if _WORLD > 1:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()


def destroy() -> None:
    global _BACKEND
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    _BACKEND = None
