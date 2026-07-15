"""Pool worker: hold one GPU slot, heartbeat, run elastic train segments."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from fleet.rendezvous import local_ip, setup_distributed
from fleet.session import (
    active_train,
    cancel_train_requested,
    shutdown_requested,
    write_ready,
    write_result,
)


def _gpu() -> tuple[str, int]:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mb = int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))
            return name, mb
    except Exception:
        pass
    try:
        line = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        ).strip().splitlines()[0]
        name, mem = [x.strip() for x in line.split(",")]
        return name, int(float(mem))
    except Exception:
        return "cpu", 0


def _mock_ttl(session_dir: str, queue: str) -> float:
    env = os.environ.get("FLEET_MOCK_TTL")
    if env:
        try:
            return float(json.loads(env).get(queue, 0) or 0)
        except Exception:
            pass
    try:
        sess = json.loads((Path(session_dir) / "session.json").read_text())
        return float((sess.get("mock_ttl") or {}).get(queue, 0) or 0)
    except Exception:
        return 0.0


def pool_worker(
    session_dir: str,
    rank: int,
    world_size: int,
    cluster: str,
    queue: str,
) -> None:
    """
    Entry for sbatch / local. `rank` is stable **slot id**; dense DDP rank
    comes from each train command when elastic participants are set.
    """
    session_dir = str(Path(session_dir).resolve())
    session_id = Path(session_dir).name
    hb = float(os.environ.get("FLEET_HEARTBEAT", "2"))
    stop = False
    # mutable cell so SIGTERM can terminate an in-flight train subprocess
    train_proc: list[subprocess.Popen[str] | None] = [None]
    started = time.time()
    ttl = _mock_ttl(session_dir, queue)

    def _stop(signum: int, frame: object) -> None:  # noqa: ARG001
        nonlocal stop
        stop = True
        # Do NOT write global SHUTDOWN — one short dying must not kill long.
        print(f"[worker] slot={rank} signal {signum}", flush=True)
        proc = train_proc[0]
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _stop)
        except Exception:
            pass
    if hasattr(signal, "SIGUSR1"):
        try:
            signal.signal(signal.SIGUSR1, _stop)
        except Exception:
            pass

    hostname = socket.gethostname()
    ip = local_ip()
    gpu_name, vram_mb = _gpu()
    print(
        f"[worker] slot={rank}/{world_size} queue={queue} host={hostname} "
        f"ip={ip} gpu={gpu_name} mock_ttl={ttl or '-'}",
        flush=True,
    )

    last_train: str | None = None
    while not stop and not shutdown_requested(session_id):
        if ttl > 0 and time.time() - started >= ttl:
            print(f"[worker] slot={rank} mock walltime {ttl}s reached — exit", flush=True)
            break

        write_ready(
            session_id,
            rank,
            {
                "rank": rank,
                "slot": rank,
                "queue": queue,
                "cluster": cluster,
                "hostname": hostname,
                "ip": ip,
                "gpu_name": gpu_name,
                "vram_mb": vram_mb,
                "pid": os.getpid(),
                "ts": time.time(),
            },
        )
        cmd = active_train(session_id)
        if cmd and cmd["train_id"] != last_train:
            tid = cmd["train_id"]
            participants = cmd.get("participants")
            if participants:
                mapping = {int(p["slot"]): int(p["dense_rank"]) for p in participants}
                if rank not in mapping:
                    print(f"[worker] slot={rank} skip train={tid} (not a participant)", flush=True)
                    last_train = tid
                    time.sleep(hb)
                    continue
                dense_rank = mapping[rank]
                train_world = len(participants)
            else:
                dense_rank = rank
                train_world = int(cmd.get("world_size") or world_size)

            print(
                f"[worker] slot={rank} train={tid} dense_rank={dense_rank}/{train_world} "
                f"seg={cmd.get('segment', 0)}",
                flush=True,
            )
            try:
                rc, out, err = _run_train(
                    session_dir,
                    session_id,
                    slot=rank,
                    dense_rank=dense_rank,
                    train_world=train_world,
                    queue=queue,
                    cmd=cmd,
                    should_stop=lambda: stop
                    or (ttl > 0 and time.time() - started >= ttl),
                    train_proc=train_proc,
                )
            except Exception as exc:
                traceback.print_exc()
                rc, out, err = 1, "", str(exc)
            finally:
                train_proc[0] = None
            write_result(session_id, tid, dense_rank, rc, out, err)
            last_train = tid
            print(f"[worker] slot={rank} train={tid} rc={rc}", flush=True)
            if stop:
                break
        time.sleep(hb)

    print(f"[worker] slot={rank} exit", flush=True)


def _run_train(
    session_dir: str,
    session_id: str,
    *,
    slot: int,
    dense_rank: int,
    train_world: int,
    queue: str,
    cmd: dict,
    should_stop: Callable[[], bool] | None = None,
    train_proc: list | None = None,
) -> tuple[int, str, str]:
    tid = cmd["train_id"]
    rdzv = Path(session_dir) / "rendezvous" / tid
    if dense_rank == 0:
        rdzv.mkdir(parents=True, exist_ok=True)

    info = setup_distributed(
        rdzv_dir=rdzv,
        rank=dense_rank,
        world_size=train_world,
        master_port=int(cmd.get("master_port") or 29500),
        backend=str(cmd.get("backend") or "auto"),
        timeout_s=float(os.environ.get("FLEET_RDZV_TIMEOUT", "600")),
    )
    print(f"[worker] slot={slot} rdzv backend={info['backend']}", flush=True)

    env = os.environ.copy()
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "FLEET_BACKEND"):
        if k in os.environ:
            env[k] = os.environ[k]
    env["FLEET_TRAIN_ID"] = tid
    env["FLEET_SESSION"] = session_id
    env["FLEET_SLOT"] = str(slot)
    env["FLEET_QUEUE"] = queue
    env["FLEET_SEGMENT"] = str(cmd.get("segment", 0))
    env["FLEET_ELASTIC"] = "1" if cmd.get("participants") is not None else "0"
    for k, v in (cmd.get("extra_env") or {}).items():
        env[str(k)] = str(v)

    proc: subprocess.Popen[str] = subprocess.Popen(
        list(cmd["argv"]),
        cwd=cmd.get("cwd") or os.getcwd(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if train_proc is not None:
        train_proc[0] = proc

    while proc.poll() is None:
        stop_req = bool(should_stop and should_stop())
        if stop_req or cancel_train_requested(session_id, tid) or shutdown_requested(session_id):
            print(f"[worker] slot={slot} cancelling train={tid}", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        time.sleep(0.5)
    stdout, stderr = proc.communicate()
    if stdout:
        sys.stdout.write(stdout)
    if stderr:
        sys.stderr.write(stderr)

    def tail(s: str) -> str:
        return s if len(s) <= 4000 else s[-4000:]

    rc = proc.returncode if proc.returncode is not None else 1
    return rc, tail(stdout or ""), tail(stderr or "")
