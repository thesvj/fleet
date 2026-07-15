"""Core data types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class QueueSpec:
    name: str
    max_jobs: int
    walltime: str  # SLURM --time
    gres: str
    cpus: int
    mem: str
    token_cost: float
    partition: str | None = None
    qos: str | None = None
    nodelist: str | None = None
    modules: tuple[str, ...] = ()

    def partition_name(self) -> str:
        return self.partition or self.name

    def qos_name(self) -> str:
        return self.qos or self.name


@dataclass(frozen=True)
class ClusterSpec:
    name: str
    account: str
    login_host: str
    user: str
    queues: dict[str, QueueSpec]
    env_prefix: str = ""
    notes: str = ""

    def max_world(self) -> int:
        return sum(q.max_jobs for q in self.queues.values())


@dataclass
class RankSlot:
    """One durable slot in the pool (queue type + generation of jobs)."""

    rank: int  # stable slot id 0..world-1
    queue: str
    job_id: str | None = None
    log_dir: str | None = None
    state: str = "pending"  # pending | ready | dead
    hostname: str | None = None
    ip: str | None = None
    gpu_name: str | None = None
    vram_mb: int | None = None
    generation: int = 0  # how many jobs have been launched in this slot
    job_history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RankSlot:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        known.setdefault("job_history", [])
        known.setdefault("generation", 0)
        return cls(**known)


@dataclass
class Session:
    session_id: str
    cluster: str
    fill: dict[str, int]
    world_size: int
    created_at: float
    ranks: list[RankSlot] = field(default_factory=list)
    master_port: int = 29500
    # Keep short/medium refilled while at least one anchor-queue slot is alive.
    replenish: bool = False
    replenish_anchor: str = "long"
    # Mock-only: seconds until a worker self-exits (simulate walltime). 0 = never.
    mock_ttl: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "cluster": self.cluster,
            "fill": self.fill,
            "world_size": self.world_size,
            "created_at": self.created_at,
            "master_port": self.master_port,
            "replenish": self.replenish,
            "replenish_anchor": self.replenish_anchor,
            "mock_ttl": self.mock_ttl,
            "ranks": [r.to_dict() for r in self.ranks],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Session:
        return cls(
            session_id=d["session_id"],
            cluster=d["cluster"],
            fill=d["fill"],
            world_size=int(d["world_size"]),
            created_at=float(d["created_at"]),
            master_port=int(d.get("master_port", 29500)),
            replenish=bool(d.get("replenish", False)),
            replenish_anchor=str(d.get("replenish_anchor", "long")),
            mock_ttl={str(k): float(v) for k, v in (d.get("mock_ttl") or {}).items()},
            ranks=[RankSlot.from_dict(r) for r in d.get("ranks", [])],
        )
