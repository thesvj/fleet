"""Battle tests: full mock pool lifecycle (no SLURM, no tokens)."""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest

from fleet.pool import down, plan, status, train, up, wait_ready
from fleet.session import ready_ranks


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def test_plan_mock(fleet_home):
    p = plan("mock", "short:2,medium:1")
    assert p["world_size"] == 3
    assert p["est_tokens"] == 0.0
    assert p["ranks"][0]["queue"] == "medium"


def test_up_status_train_down(fleet_home, root: Path):
    sess = up(
        cluster_name="mock",
        fill_spec="short:2",
        wait=True,
        timeout_s=30,
    )
    assert sess.world_size == 2
    st = status("mock", sess.session_id)
    assert st["ready"] == 2
    assert all(r["state"] == "ready" for r in st["ranks"])

    smoke = root / "examples" / "smoke_env.py"
    res = train(
        cluster_name="mock",
        argv=["python", str(smoke)],
        cwd=str(root),
        backend="auto",
        session_id=sess.session_id,
        timeout_s=60,
    )
    assert res["ok"] is True
    assert len(res["results"]) == 2
    assert all(r["returncode"] == 0 for r in res["results"])

    # second train on same pool
    res2 = train(
        cluster_name="mock",
        argv=["python", str(smoke)],
        cwd=str(root),
        session_id=sess.session_id,
        timeout_s=60,
    )
    assert res2["ok"] is True

    pids = []
    for r in sess.ranks:
        jid = r.job_id or ""
        if jid.startswith("local-"):
            pids.append(int(jid.split("-", 1)[1]))

    down("mock", sess.session_id)
    time.sleep(0.5)
    for pid in pids:
        # workers should be gone (or dying)
        deadline = time.time() + 5
        while time.time() < deadline and _alive(pid):
            time.sleep(0.1)
        assert not _alive(pid), f"worker pid {pid} still alive after down"


def test_train_before_ready_fails(fleet_home, root: Path):
    sess = up(
        cluster_name="mock",
        fill_spec="short:2",
        wait=False,
        timeout_s=5,
    )
    # may race; if already ready, skip assertion path
    ready = ready_ranks(sess.session_id, sess.world_size)
    if len(ready) < sess.world_size:
        with pytest.raises(RuntimeError, match="not ready"):
            train(
                cluster_name="mock",
                argv=["python", str(root / "examples" / "smoke_env.py")],
                cwd=str(root),
                session_id=sess.session_id,
                timeout_s=5,
            )
    wait_ready("mock", sess.session_id, timeout_s=30)
    down("mock", sess.session_id)


def test_bad_fill_rejected(fleet_home):
    with pytest.raises(ValueError):
        up(cluster_name="mock", fill_spec="short:99", wait=False)


def test_world_one(fleet_home, root: Path):
    sess = up(cluster_name="mock", fill_spec="long:1", wait=True, timeout_s=20)
    assert sess.world_size == 1
    res = train(
        cluster_name="mock",
        argv=["python", str(root / "examples" / "smoke_env.py")],
        cwd=str(root),
        session_id=sess.session_id,
        timeout_s=30,
    )
    assert res["ok"]
    down("mock", sess.session_id)


def test_failing_train_command(fleet_home, root: Path):
    sess = up(cluster_name="mock", fill_spec="short:2", wait=True, timeout_s=20)
    res = train(
        cluster_name="mock",
        argv=["python", "-c", "import sys; sys.exit(7)"],
        cwd=str(root),
        session_id=sess.session_id,
        timeout_s=30,
    )
    assert res["ok"] is False
    assert all(r["returncode"] == 7 for r in res["results"])
    down("mock", sess.session_id)


@pytest.mark.torch
def test_ddp_smoke_if_torch(fleet_home, root: Path):
    torch = pytest.importorskip("torch")
    _ = torch
    sess = up(cluster_name="mock", fill_spec="short:2", wait=True, timeout_s=30)
    res = train(
        cluster_name="mock",
        argv=["python", str(root / "examples" / "smoke_ddp.py")],
        cwd=str(root),
        backend="gloo",
        session_id=sess.session_id,
        timeout_s=120,
    )
    assert res["ok"] is True
    down("mock", sess.session_id)
