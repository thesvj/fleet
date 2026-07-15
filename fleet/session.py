"""Shared-filesystem session state (ready heartbeats, train commands, results)."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from fleet.schema import Session


def fleet_home() -> Path:
    raw = os.environ.get("FLEET_HOME")
    p = Path(raw).expanduser() if raw else Path.home() / ".fleet"
    p.mkdir(parents=True, exist_ok=True)
    return p


def session_path(session_id: str) -> Path:
    d = fleet_home() / "sessions" / session_id
    for sub in ("ready", "meta", "commands", "results", "rendezvous", "logs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def active_path(cluster: str) -> Path:
    return fleet_home() / f"active_{cluster}.session"


def save(sess: Session) -> Path:
    path = session_path(sess.session_id) / "session.json"
    path.write_text(json.dumps(sess.to_dict(), indent=2) + "\n")
    active_path(sess.cluster).write_text(sess.session_id + "\n")
    return path


def load(session_id: str) -> Session:
    path = session_path(session_id) / "session.json"
    if not path.exists():
        raise FileNotFoundError(f"session not found: {session_id}")
    return Session.from_dict(json.loads(path.read_text()))


def resolve_id(cluster: str, session_id: str | None = None) -> str:
    if session_id:
        return session_id
    ptr = active_path(cluster)
    if ptr.exists():
        sid = ptr.read_text().strip()
        if sid:
            return sid
    raise FileNotFoundError(
        f"no active session for {cluster!r}; run: fleet start -c {cluster} -f ..."
    )


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def write_ready(session_id: str, rank: int, meta: dict[str, Any]) -> None:
    d = session_path(session_id)
    (d / "meta" / f"{rank}.json").write_text(json.dumps(meta, indent=2) + "\n")
    (d / "ready" / str(rank)).write_text(f"{time.time():.3f}\n")


def ready_ranks(session_id: str, world_size: int, max_age_s: float = 90.0) -> list[int]:
    d = session_path(session_id) / "ready"
    now = time.time()
    out: list[int] = []
    for r in range(world_size):
        p = d / str(r)
        if not p.exists():
            continue
        try:
            ts = float(p.read_text().strip())
        except ValueError:
            ts = now
        if now - ts <= max_age_s:
            out.append(r)
    return out


def read_meta(session_id: str, rank: int) -> dict[str, Any] | None:
    p = session_path(session_id) / "meta" / f"{rank}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def request_shutdown(session_id: str) -> None:
    (session_path(session_id) / "SHUTDOWN").write_text(f"{time.time():.3f}\n")


def shutdown_requested(session_id: str) -> bool:
    return (session_path(session_id) / "SHUTDOWN").exists()


def clear_shutdown(session_id: str) -> None:
    (session_path(session_id) / "SHUTDOWN").unlink(missing_ok=True)


def write_train(
    session_id: str,
    *,
    argv: list[str],
    cwd: str,
    backend: str,
    master_port: int,
    participants: list[dict[str, int]] | None = None,
    segment: int = 0,
    world_size: int | None = None,
    extra_env: dict[str, str] | None = None,
) -> str:
    """
    participants: [{slot, dense_rank}, ...] for elastic segments.
    If None, all slots 0..world-1 participate with dense_rank=slot.
    """
    train_id = "train-" + uuid.uuid4().hex[:10]
    payload = {
        "train_id": train_id,
        "argv": argv,
        "cwd": cwd,
        "backend": backend,
        "master_port": master_port,
        "created_at": time.time(),
        "segment": segment,
        "participants": participants,
        "world_size": world_size,
        "extra_env": extra_env or {},
    }
    d = session_path(session_id) / "commands"
    (d / f"{train_id}.json").write_text(json.dumps(payload, indent=2) + "\n")
    (d / "ACTIVE").write_text(train_id + "\n")
    return train_id


def active_train(session_id: str) -> dict[str, Any] | None:
    ptr = session_path(session_id) / "commands" / "ACTIVE"
    if not ptr.exists():
        return None
    train_id = ptr.read_text().strip()
    path = session_path(session_id) / "commands" / f"{train_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def clear_train(session_id: str) -> None:
    (session_path(session_id) / "commands" / "ACTIVE").unlink(missing_ok=True)


def write_result(
    session_id: str,
    train_id: str,
    rank: int,
    returncode: int,
    stdout_tail: str = "",
    stderr_tail: str = "",
) -> None:
    d = session_path(session_id) / "results" / train_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{rank}.json"
    # Do not overwrite a real worker result with a synthetic fill.
    if path.exists():
        return
    path.write_text(
        json.dumps(
            {
                "rank": rank,
                "returncode": returncode,
                "stdout_tail": stdout_tail[-4000:],
                "stderr_tail": stderr_tail[-4000:],
                "finished_at": time.time(),
            },
            indent=2,
        )
        + "\n"
    )


def collect_results(
    session_id: str,
    train_id: str,
    world_size: int,
    *,
    dense_ranks: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Collect result files keyed by dense rank (0..k-1 for elastic segments)."""
    d = session_path(session_id) / "results" / train_id
    out: list[dict[str, Any]] = []
    if not d.exists():
        return out
    keys = dense_ranks if dense_ranks is not None else list(range(world_size))
    for r in keys:
        p = d / f"{r}.json"
        if p.exists():
            out.append(json.loads(p.read_text()))
    return out


def clear_slot_ready(session_id: str, rank: int) -> None:
    (session_path(session_id) / "ready" / str(rank)).unlink(missing_ok=True)
    (session_path(session_id) / "meta" / f"{rank}.json").unlink(missing_ok=True)


def request_cancel_train(session_id: str, train_id: str) -> None:
    d = session_path(session_id) / "commands"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{train_id}.cancel").write_text(f"{time.time():.3f}\n")


def cancel_train_requested(session_id: str, train_id: str) -> bool:
    return (session_path(session_id) / "commands" / f"{train_id}.cancel").exists()
