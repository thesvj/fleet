"""Multi-job rendezvous via shared FS — env vars only; train script owns torch PG."""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path
from typing import Any


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("1.1.1.1", 80))
            ip = s.getsockname()[0]
            if not ip.startswith("127."):
                return ip
    except OSError:
        pass
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


def wait_file(path: Path, timeout_s: float = 600.0, poll_s: float = 0.25) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return path.read_text().strip()
        time.sleep(poll_s)
    raise TimeoutError(f"timeout waiting for {path}")


def file_barrier(
    barrier_dir: Path,
    rank: int,
    world_size: int,
    timeout_s: float = 600.0,
    poll_s: float = 0.25,
) -> None:
    barrier_dir.mkdir(parents=True, exist_ok=True)
    (barrier_dir / str(rank)).write_text(f"{time.time():.3f}\n")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if all((barrier_dir / str(r)).exists() for r in range(world_size)):
            return
        time.sleep(poll_s)
    have = sum(1 for r in range(world_size) if (barrier_dir / str(r)).exists())
    raise TimeoutError(f"barrier {barrier_dir}: {have}/{world_size}")


def setup_distributed(
    *,
    rdzv_dir: Path,
    rank: int,
    world_size: int,
    master_port: int,
    backend: str = "auto",
    timeout_s: float = 1800.0,
) -> dict[str, Any]:
    """
    Rank-0 publishes master addr/port; all ranks file-barrier; set RANK/WORLD_SIZE/MASTER_*.

    Does **not** call torch.distributed — the train subprocess owns the process group
    (via fleet.init() or torchrun-style env).
    """
    rdzv_dir.mkdir(parents=True, exist_ok=True)
    addr_p = rdzv_dir / "master_addr"
    port_p = rdzv_dir / "master_port"

    if rank == 0:
        addr_p.write_text(local_ip() + "\n")
        port_p.write_text(f"{master_port}\n")
    else:
        wait_file(addr_p, timeout_s=timeout_s)

    master_addr = addr_p.read_text().strip()
    master_port_s = port_p.read_text().strip() if port_p.exists() else str(master_port)
    file_barrier(rdzv_dir / "barrier", rank, world_size, timeout_s=timeout_s)

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port_s
    os.environ["FLEET_BACKEND"] = backend

    return {
        "rank": rank,
        "world_size": world_size,
        "master_addr": master_addr,
        "master_port": int(master_port_s),
        "backend": "none" if world_size == 1 else "env-only",
    }
