#!/usr/bin/env python3
"""No-torch smoke: verify RANK / WORLD_SIZE / MASTER_* across the pool."""

from __future__ import annotations

import os
import socket
import sys


def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    master = os.environ.get("MASTER_ADDR", "")
    port = os.environ.get("MASTER_PORT", "")
    host = socket.gethostname()
    print(
        f"[smoke_env] host={host} rank={rank}/{world} master={master}:{port}",
        flush=True,
    )
    if rank < 0 or rank >= world:
        print(f"[smoke_env] bad rank {rank} for world {world}", file=sys.stderr)
        return 2
    if world > 1 and not master:
        print("[smoke_env] missing MASTER_ADDR", file=sys.stderr)
        return 2
    print(f"[smoke_env] PASS rank={rank}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
