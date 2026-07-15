#!/usr/bin/env python3
"""
Stdlib simulation runner — no pytest/network required.

  PYTHONPATH=. python tests/run_sim.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Fail(Exception):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise Fail(msg)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def test_squeue_helpers() -> None:
    from fleet.executor import is_dead_slurm_state, local_job_view

    check(is_dead_slurm_state("FAILED"), "FAILED dead")
    check(is_dead_slurm_state("F"), "F dead")
    check(not is_dead_slurm_state("RUNNING"), "R live")
    view = local_job_view(f"local-{os.getpid()}")
    check(view["state"] == "RUNNING", "local running")


def test_config() -> None:
    from fleet.config import (
        get_cluster,
        parse_fill,
        rank_plan,
        token_cost,
        walltime_minutes,
    )

    c = get_cluster("omni")
    fill = parse_fill("all", c)
    check(fill == {"short": 3, "medium": 2, "long": 1}, "fill all")
    check(abs(token_cost(c, fill) - 2.3) < 1e-9, "token cost")
    check(rank_plan(fill)[0] == "long", "rank0 long")
    check(walltime_minutes("1-00:00:00") == 1440, "walltime")
    try:
        parse_fill("long:2", c)
        raise Fail("expected cap error")
    except ValueError:
        pass


def test_session(home: Path) -> None:
    os.environ["FLEET_HOME"] = str(home)
    from fleet.schema import Session
    from fleet.session import load, new_id, ready_ranks, save, write_ready

    sid = new_id()
    save(
        Session(
            session_id=sid,
            cluster="mock",
            fill={"short": 1},
            world_size=1,
            created_at=time.time(),
        )
    )
    check(load(sid).session_id == sid, "session load")
    write_ready(sid, 0, {"rank": 0})
    check(ready_ranks(sid, 1) == [0], "ready")


def test_e2e(home: Path) -> None:
    os.environ["FLEET_HOME"] = str(home)
    os.chdir(ROOT)
    # re-import pool against this home
    import importlib

    import fleet.pool as pool
    import fleet.session as session

    importlib.reload(session)
    importlib.reload(pool)

    from fleet.pool import down, status, train, up

    sess = up(cluster_name="mock", fill_spec="short:2", wait=True, timeout_s=30)
    check(sess.world_size == 2, "world 2")
    st = status("mock", sess.session_id)
    check(st["ready"] == 2, f"ready got {st['ready']}")

    smoke = ROOT / "examples" / "smoke_env.py"
    res = train(
        cluster_name="mock",
        argv=[sys.executable, str(smoke)],
        cwd=str(ROOT),
        session_id=sess.session_id,
        timeout_s=60,
    )
    check(res["ok"] is True, f"train1 not ok: {res}")

    res2 = train(
        cluster_name="mock",
        argv=[sys.executable, str(smoke)],
        cwd=str(ROOT),
        session_id=sess.session_id,
        timeout_s=60,
    )
    check(res2["ok"] is True, "train2 not ok")

    # failing command
    res3 = train(
        cluster_name="mock",
        argv=[sys.executable, "-c", "import sys; sys.exit(7)"],
        cwd=str(ROOT),
        session_id=sess.session_id,
        timeout_s=30,
    )
    check(res3["ok"] is False, "expected fail")
    check(all(r["returncode"] == 7 for r in res3["results"]), "rc 7")

    pids = [
        int(r.job_id.split("-", 1)[1])
        for r in sess.ranks
        if r.job_id and r.job_id.startswith("local-")
    ]
    down("mock", sess.session_id)
    time.sleep(0.8)
    for pid in pids:
        deadline = time.time() + 5
        while time.time() < deadline and alive(pid):
            time.sleep(0.1)
        check(not alive(pid), f"pid {pid} still alive")


def test_world_one(home: Path) -> None:
    os.environ["FLEET_HOME"] = str(home)
    os.chdir(ROOT)
    import importlib

    import fleet.pool as pool
    import fleet.session as session

    importlib.reload(session)
    importlib.reload(pool)
    from fleet.pool import down, train, up

    sess = up(cluster_name="mock", fill_spec="long:1", wait=True, timeout_s=20)
    check(sess.world_size == 1, "world 1")
    res = train(
        cluster_name="mock",
        argv=[sys.executable, str(ROOT / "examples" / "smoke_env.py")],
        cwd=str(ROOT),
        session_id=sess.session_id,
        timeout_s=30,
    )
    check(res["ok"], "world1 train")
    down("mock", sess.session_id)


def main() -> int:
    tests = [
        ("config", "unit"),
        ("squeue_helpers", "unit"),
        ("session", "home"),
        ("e2e", "home"),
        ("world_one", "home"),
        ("status_slurm_fields", "home"),
        ("replenish_and_elastic", "home"),
    ]
    failed = 0
    print("=== fleet simulation suite ===")
    for name, kind in tests:
        print(f"— {name} …", end=" ", flush=True)
        try:
            if name == "config":
                test_config()
            elif name == "squeue_helpers":
                test_squeue_helpers()
            else:
                home = Path(tempfile.mkdtemp(prefix=f"fleet-{name}-"))
                try:
                    if name == "session":
                        test_session(home)
                    elif name == "e2e":
                        test_e2e(home)
                    elif name == "world_one":
                        test_world_one(home)
                    elif name == "status_slurm_fields":
                        test_status_slurm_fields(home)
                    elif name == "replenish_and_elastic":
                        test_replenish_and_elastic(home)
                finally:
                    shutil.rmtree(home, ignore_errors=True)
            print("OK")
        except Exception as exc:
            failed += 1
            print("FAIL")
            traceback.print_exc()
            print(f"  {exc}")
    print(f"=== {len(tests) - failed}/{len(tests)} passed ===")
    return 1 if failed else 0


def test_replenish_and_elastic(home: Path) -> None:
    """short dies via mock TTL; replenish + elastic train on remaining."""
    os.environ["FLEET_HOME"] = str(home)
    os.chdir(ROOT)
    import importlib

    import fleet.pool as pool
    import fleet.session as session

    importlib.reload(session)
    importlib.reload(pool)
    from fleet.pool import down, replenish_once, status, train, up, wait_ready

    sess = up(
        cluster_name="mock",
        fill_spec="long:1,medium:1,short:1",
        wait=True,
        timeout_s=30,
        replenish=True,
        mock_ttl={"short": 3.0, "medium": 20.0, "long": 60.0},
    )
    check(sess.world_size == 3, "world 3")
    # wait for short mock walltime (3s) + process exit
    time.sleep(5.0)
    # poll a few times — local PID must be gone before refill
    n = 0
    for _ in range(8):
        n = replenish_once("mock", sess.session_id)
        if n >= 1:
            break
        time.sleep(0.5)
    check(n >= 1, f"expected replenish ≥1 got {n}")
    wait_ready("mock", sess.session_id, timeout_s=20, min_world=2)
    st = status("mock", sess.session_id)
    short = next(r for r in st["ranks"] if r["queue"] == "short")
    check(short["generation"] >= 2, f"short gen={short['generation']}")

    # elastic: may run on 2 or 3 ready slots
    res = train(
        cluster_name="mock",
        argv=[sys.executable, str(ROOT / "examples" / "smoke_env.py")],
        cwd=str(ROOT),
        session_id=sess.session_id,
        timeout_s=30,
        elastic=True,
        min_world=1,
        segment=0,
    )
    check(res["ok"] is True, f"elastic train {res}")
    down("mock", sess.session_id)


def test_status_slurm_fields(home: Path) -> None:
    os.environ["FLEET_HOME"] = str(home)
    os.chdir(ROOT)
    import importlib

    import fleet.pool as pool
    import fleet.session as session

    importlib.reload(session)
    importlib.reload(pool)
    from fleet.pool import down, squeue_status, status, up

    sess = up(cluster_name="mock", fill_spec="short:1", wait=True, timeout_s=20)
    st = status("mock", sess.session_id)
    row = st["ranks"][0]
    check("slurm_state" in row, "slurm_state field")
    check(row["state"] == "ready", "fleet ready")
    check(row["slurm_compact"] in ("R", "CD"), f"compact={row['slurm_compact']}")
    sq = squeue_status("mock", sess.session_id)
    check(len(sq["squeue_session"]) == 1, "squeue session lines")
    down("mock", sess.session_id)


if __name__ == "__main__":
    raise SystemExit(main())
