"""Cluster registry and fill parsing (IIITD OMNI / Precision style)."""

from __future__ import annotations

import os
from pathlib import Path

from fleet.schema import ClusterSpec, QueueSpec

# ---------------------------------------------------------------------------
# Local secrets: copy .env.example → .env (gitignored) and edit.
# Real hosts/accounts never belong in the public tree.
# ---------------------------------------------------------------------------

# Published placeholders only (not real campus hosts).
_DUMMY_OMNI_HOST = "10.0.0.1"
_DUMMY_PRECISION_HOST = "10.0.0.2"
_DUMMY_ACCOUNT = "lab"


def _load_dotenv() -> None:
    """Load KEY=VAL from .env into os.environ (does not override existing)."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = val
        break  # first found .env wins


_load_dotenv()


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


_DEFAULT_USER = _env("FLEET_USER") or _env("USER") or "user"
_ACCOUNT = _env("FLEET_ACCOUNT", _DUMMY_ACCOUNT)

OMNI = ClusterSpec(
    name="omni",
    account=_ACCOUNT,
    login_host=_env("FLEET_OMNI_HOST", _DUMMY_OMNI_HOST),
    user=_DEFAULT_USER,
    env_prefix=(
        "export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf; "
        "export PATH=$PATH:/cm/shared/apps/slurm/current/bin; "
    ),
    notes="OMNI: short=MIG 3g.71gb; medium/long=full GPU; CUDA 12.8 only.",
    queues={
        "short": QueueSpec(
            "short", 3, "6:00:00", "gpu:3g.71gb:1", 10, "64G", 0.1,
            nodelist="dgxh200", modules=("cuda12.8/toolkit/12.8.1",),
        ),
        "medium": QueueSpec(
            "medium", 2, "1-00:00:00", "gpu:1", 16, "256G", 0.5,
            modules=("cuda12.8/toolkit/12.8.1",),
        ),
        "long": QueueSpec(
            "long", 1, "3-00:00:00", "gpu:1", 20, "512G", 1.0,
            modules=("cuda12.8/toolkit/12.8.1",),
        ),
    },
)

PRECISION = ClusterSpec(
    name="precision",
    account=_ACCOUNT,
    login_host=_env("FLEET_PRECISION_HOST", _DUMMY_PRECISION_HOST),
    user=_DEFAULT_USER,
    notes="Precision: short=MIG 3g.40gb; medium=H100/A100; long=H200; CUDA 12.4.",
    queues={
        "short": QueueSpec(
            "short", 3, "2:00:00", "gpu:3g.40gb:1", 10, "64G", 0.1,
            nodelist="gpu01", modules=("cuda-12.4",),
        ),
        "medium": QueueSpec(
            "medium", 2, "12:00:00", "gpu:1", 16, "256G", 0.5,
            modules=("cuda-12.4",),
        ),
        "long": QueueSpec(
            "long", 1, "3-00:00:00", "gpu:1", 20, "256G", 1.0,
            modules=("cuda-12.4",),
        ),
    },
)

MOCK = ClusterSpec(
    name="mock",
    account="local",
    login_host="localhost",
    user="local",
    notes="Local multi-process simulation. Zero tokens. Use before any real cluster submit.",
    queues={
        "short": QueueSpec("short", 3, "1:00:00", "mock:1", 2, "4G", 0.0),
        "medium": QueueSpec("medium", 2, "1:00:00", "mock:1", 2, "4G", 0.0),
        "long": QueueSpec("long", 1, "1:00:00", "mock:1", 2, "4G", 0.0),
    },
)

CLUSTERS: dict[str, ClusterSpec] = {
    "omni": OMNI,
    "precision": PRECISION,
    "mock": MOCK,
}


def get_cluster(name: str) -> ClusterSpec:
    key = name.lower().strip()
    if key not in CLUSTERS:
        raise KeyError(f"unknown cluster {name!r}; known: {', '.join(sorted(CLUSTERS))}")
    return CLUSTERS[key]


def parse_fill(spec: str | None, cluster: ClusterSpec) -> dict[str, int]:
    """Parse 'short:3,medium:2' or 'all' into per-queue counts. Enforces MaxJobsPA."""
    if not spec or spec in ("all", "full", "max"):
        return {name: q.max_jobs for name, q in cluster.queues.items()}

    out = {name: 0 for name in cluster.queues}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, n_s = part.split(":", 1)
            n = int(n_s)
        else:
            name, n = part, cluster.queues[part].max_jobs
        name = name.strip()
        if name not in cluster.queues:
            raise ValueError(f"queue {name!r} not on {cluster.name}")
        cap = cluster.queues[name].max_jobs
        if n < 0 or n > cap:
            raise ValueError(f"{cluster.name}/{name}: requested {n}, cap is {cap}")
        out[name] = n
    return out


def token_cost(cluster: ClusterSpec, fill: dict[str, int]) -> float:
    return sum(cluster.queues[q].token_cost * n for q, n in fill.items())


def rank_plan(fill: dict[str, int]) -> list[str]:
    """Stable rank→queue map. Prefer long → medium → short so rank0 is most stable."""
    order = ("long", "medium", "short")
    ranks: list[str] = []
    for q in order:
        ranks.extend([q] * fill.get(q, 0))
    for q, n in fill.items():
        if q not in order:
            ranks.extend([q] * n)
    return ranks


def walltime_minutes(walltime: str) -> int:
    """SLURM time → minutes (ceil)."""
    s = walltime.strip()
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(x) for x in s.split(":")]
    if len(parts) == 1:
        sec = days * 86400 + parts[0] * 60
    elif len(parts) == 2:
        sec = days * 86400 + parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        sec = days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    else:
        raise ValueError(f"bad walltime {walltime!r}")
    return max(1, (sec + 59) // 60)


def mem_gb(mem: str) -> int:
    m = mem.strip().upper()
    if m.endswith("G"):
        return max(1, int(float(m[:-1])))
    if m.endswith("M"):
        return max(1, int(float(m[:-1])) // 1024 or 1)
    return max(1, int(float(m)))
