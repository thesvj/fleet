"""Submit one pool worker per GPU (local / sbatch) + squeue/scancel helpers."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from fleet.config import ClusterSpec, QueueSpec

# Terminal SLURM states — job will never become a live worker.
_SLURM_DEAD = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
        "SPECIAL_EXIT",
    }
)
_SLURM_DEAD_COMPACT = frozenset({"BF", "CA", "CD", "DL", "F", "NF", "OOM", "PR", "TO", "SE"})


@dataclass
class Job:
    job_id: str
    rank: int
    queue: str
    backend: str
    log_dir: str
    _cancel: Callable[[], None]
    _proc: Any = None  # subprocess.Popen for local jobs

    def cancel(self) -> None:
        self._cancel()

    def is_running(self) -> bool:
        if self._proc is not None:
            return self._proc.poll() is None
        return local_pid_running(self.job_id)


def backend_label() -> str:
    return "local+sbatch"


def submit_worker(
    *,
    cluster: ClusterSpec,
    queue: QueueSpec,
    rank: int,
    world_size: int,
    session_dir: str,
    force_local: bool = False,
) -> Job:
    log_dir = Path(session_dir) / "logs" / f"r{rank}_{queue.name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    if force_local or cluster.name == "mock":
        return _local(session_dir, rank, world_size, cluster.name, queue.name, log_dir)
    return _sbatch(cluster, queue, session_dir, rank, world_size, log_dir)


def _local(
    session_dir: str,
    rank: int,
    world_size: int,
    cluster: str,
    queue: str,
    log_dir: Path,
) -> Job:
    root = str(Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    log_f = open(log_dir / "worker.log", "w")  # noqa: SIM115
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "from fleet.worker import pool_worker; "
                f"pool_worker({session_dir!r}, {rank}, {world_size}, {cluster!r}, {queue!r})"
            ),
        ],
        cwd=root,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def cancel() -> None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    return Job(f"local-{proc.pid}", rank, queue, "local", str(log_dir), cancel, _proc=proc)


def _sbatch(
    cluster: ClusterSpec,
    queue: QueueSpec,
    session_dir: str,
    rank: int,
    world_size: int,
    log_dir: Path,
) -> Job:
    fleet_root = str(Path(__file__).resolve().parents[1])
    py = sys.executable
    script = log_dir / "worker.sbatch"
    nodelist = f"#SBATCH --nodelist={queue.nodelist}" if queue.nodelist else ""
    mods = "\n".join(
        f"module load {m} 2>/dev/null || module add {m} 2>/dev/null || true"
        for m in queue.modules
    )
    script.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            #SBATCH -A {cluster.account}
            #SBATCH -p {queue.partition_name()}
            #SBATCH -q {queue.qos_name()}
            #SBATCH --job-name=fleet-r{rank}-{queue.name}
            #SBATCH --ntasks=1
            #SBATCH --cpus-per-task={queue.cpus}
            #SBATCH --mem={queue.mem}
            #SBATCH --gres={queue.gres}
            #SBATCH --time={queue.walltime}
            #SBATCH --signal=B:USR1@180
            #SBATCH --output={log_dir}/%j.out
            {nodelist}
            set -euo pipefail
            {mods}
            export PYTHONPATH="{fleet_root}${{PYTHONPATH:+:$PYTHONPATH}}"
            export PYTHONUNBUFFERED=1
            {py} -c "from fleet.worker import pool_worker; pool_worker({session_dir!r}, {rank}, {world_size}, {cluster.name!r}, {queue.name!r})"
            """
        )
    )
    script.chmod(0o755)
    proc = subprocess.run(
        ["bash", "-lc", f"{cluster.env_prefix}sbatch --parsable {script}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"sbatch failed:\n{proc.stdout}\n{proc.stderr}")
    jid = proc.stdout.strip().split(";")[0]

    def cancel() -> None:
        subprocess.run(
            ["bash", "-lc", f"{cluster.env_prefix}scancel {jid}"],
            check=False,
        )

    return Job(jid, rank, queue.name, "sbatch", str(log_dir), cancel)


def _run_slurm(cluster: ClusterSpec, cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", f"{cluster.env_prefix}{cmd}"],
        capture_output=True,
        text=True,
        check=False,
    )


def is_slurm_job_id(job_id: str | None) -> bool:
    return bool(job_id) and str(job_id).isdigit()


def is_dead_slurm_state(state: str | None) -> bool:
    if not state:
        return False
    base = state.strip().upper().split("+", 1)[0]
    return base in _SLURM_DEAD or base in _SLURM_DEAD_COMPACT


def squeue_jobs(cluster: ClusterSpec, job_ids: list[str]) -> dict[str, dict[str, str]]:
    """job_id → {state, state_compact, reason, node, partition, elapsed, name}."""
    numeric = [j for j in job_ids if is_slurm_job_id(j)]
    if not numeric or cluster.name == "mock":
        return {}
    fmt = "%i|%T|%t|%R|%N|%P|%M|%j"
    ids = ",".join(numeric)
    proc = _run_slurm(cluster, f"squeue -h -j {ids} -o '{fmt}' 2>/dev/null")
    out: dict[str, dict[str, str]] = {}
    if proc.returncode != 0 and not proc.stdout.strip():
        return out
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) < 8:
            continue
        jid, state, compact, reason, node, part, elapsed, name = parts[:8]
        out[jid.strip()] = {
            "job_id": jid.strip(),
            "state": state.strip(),
            "state_compact": compact.strip(),
            "reason": reason.strip(),
            "node": node.strip(),
            "partition": part.strip(),
            "elapsed": elapsed.strip(),
            "name": name.strip(),
        }
    return out


def squeue_user(cluster: ClusterSpec, user: str | None = None) -> str:
    u = user or cluster.user or os.environ.get("USER", "")
    if cluster.name == "mock":
        return "(mock cluster — no squeue)"
    proc = _run_slurm(
        cluster,
        f"squeue -u {u} -o '%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R' 2>/dev/null",
    )
    if proc.returncode != 0 and not proc.stdout.strip():
        err = (proc.stderr or "").strip() or "squeue failed or not on PATH"
        return f"(squeue unavailable: {err})"
    return proc.stdout.rstrip() or f"(no jobs for user {u})"


def scancel_job(cluster: ClusterSpec, job_id: str) -> None:
    if is_slurm_job_id(job_id):
        _run_slurm(cluster, f"scancel {job_id}")


def local_pid_running(job_id: str | None) -> bool:
    """True if local-PID job is a live non-zombie process."""
    if not job_id or not str(job_id).startswith("local-"):
        return False
    try:
        pid = int(str(job_id).split("-", 1)[1])
    except ValueError:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("State:"):
                code = line.split()[1] if len(line.split()) > 1 else ""
                return code not in ("Z", "X")
    except OSError:
        pass
    return True


def local_job_view(job_id: str | None) -> dict[str, str]:
    """Synthetic squeue-like row for local/mock workers."""
    base = {
        "job_id": job_id or "-",
        "state": "UNKNOWN",
        "state_compact": "?",
        "reason": "-",
        "node": "-",
        "partition": "local",
        "elapsed": "-",
        "name": "fleet",
    }
    if not job_id:
        return base
    if job_id.startswith("local-"):
        running = local_pid_running(job_id)
        return {
            **base,
            "state": "RUNNING" if running else "COMPLETED",
            "state_compact": "R" if running else "CD",
            "reason": "local-process" if running else "exited",
            "node": "localhost",
            "name": "fleet-local",
        }
    return {**base, "reason": "not-in-squeue", "partition": "-"}
