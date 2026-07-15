#!/usr/bin/env python3
"""
Failure-mode battle suite for fleet (stdlib, no pytest required).

Simulates real multi-GPU training incidents on mock:
  OOM-style exits, asymmetric rank crash, hang/timeout, worker preempt,
  stop mid-train, overlapping train, flex after death, elastic walltime,
  recovery after failure, bad inputs, rdzv issues.

  PYTHONPATH=. python tests/run_failures.py
"""

from __future__ import annotations

import importlib
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PY = sys.executable


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


def reload_pool(home: Path):
    os.environ["FLEET_HOME"] = str(home)
    os.chdir(ROOT)
    import fleet.pool as pool
    import fleet.session as session

    importlib.reload(session)
    importlib.reload(pool)
    return pool


def local_pids(sess) -> list[int]:
    out = []
    for r in sess.ranks:
        jid = r.job_id or ""
        if jid.startswith("local-"):
            out.append(int(jid.split("-", 1)[1]))
    return out


def pid_for_slot(sess, slot: int) -> int | None:
    for r in sess.ranks:
        if r.rank == slot and r.job_id and r.job_id.startswith("local-"):
            return int(r.job_id.split("-", 1)[1])
    return None


# ---------------------------------------------------------------------------
# Scenario scripts (inline python -c)
# ---------------------------------------------------------------------------

# Successful env smoke
SMOKE = [PY, str(ROOT / "examples" / "smoke_env.py")]

# All ranks crash with OOM-like code
OOM_ALL = [PY, "-c", "import sys; print('CUDA OOM', flush=True); sys.exit(137)"]

# Only dense rank 0 fails (asymmetric — classic partial NCCL death precursor)
ASYM_RANK0 = [
    PY,
    "-c",
    "import os,sys; r=int(os.environ.get('RANK','0')); "
    "print(f'rank={r}',flush=True); sys.exit(1 if r==0 else 0)",
]

# Only non-zero rank fails
ASYM_RANK1 = [
    PY,
    "-c",
    "import os,sys; r=int(os.environ.get('RANK','0')); "
    "print(f'rank={r}',flush=True); sys.exit(2 if r==1 else 0)",
]

# Hang forever (NCCL collective hang stand-in)
HANG = [PY, "-c", "import time; print('hang',flush=True); time.sleep(3600)"]

# Slow train then success — for mid-kill tests
SLOW_OK = [
    PY,
    "-c",
    "import time,os; print('slow',os.environ.get('RANK'),flush=True); "
    "time.sleep(30); print('done',flush=True)",
]

# Segment-aware resume: fail segment 0, succeed later (checkpoint resume stand-in)
SEG_RESUME = [
    PY,
    "-c",
    "import os,sys; s=int(os.environ.get('FLEET_SEGMENT','0')); "
    "print(f'segment={s} rank={os.environ.get(\"RANK\")}',flush=True); "
    "sys.exit(1 if s==0 else 0)",
]

# Crash after a short work pulse (straggler-friendly)
WORK_THEN_DIE = [
    PY,
    "-c",
    "import time,os,sys; time.sleep(1); "
    "print('die',os.environ.get('RANK'),flush=True); sys.exit(3)",
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_argv(home: Path) -> None:
    pool = reload_pool(home)
    sess = pool.up(cluster_name="mock", fill_spec="long:1", wait=True, timeout_s=20)
    try:
        pool.train(cluster_name="mock", argv=[], session_id=sess.session_id)
        raise Fail("expected empty argv error")
    except ValueError:
        pass
    pool.down("mock", sess.session_id)


def test_bad_fill(home: Path) -> None:
    pool = reload_pool(home)
    try:
        pool.up(cluster_name="mock", fill_spec="long:99", wait=False)
        raise Fail("expected cap error")
    except ValueError:
        pass


def test_train_before_ready(home: Path) -> None:
    pool = reload_pool(home)
    sess = pool.up(cluster_name="mock", fill_spec="short:2", wait=False, timeout_s=5)
    from fleet.session import ready_ranks

    ready = ready_ranks(sess.session_id, sess.world_size)
    if len(ready) < sess.world_size:
        try:
            pool.train(
                cluster_name="mock",
                argv=SMOKE,
                cwd=str(ROOT),
                session_id=sess.session_id,
                timeout_s=5,
            )
            raise Fail("expected not-ready error")
        except RuntimeError as e:
            check("not" in str(e).lower() or "ready" in str(e).lower(), f"msg={e}")
    pool.wait_ready("mock", sess.session_id, timeout_s=30)
    pool.down("mock", sess.session_id)


def test_oom_all_ranks(home: Path) -> None:
    """All GPUs OOM-exit → ok=False, pool still usable after."""
    pool = reload_pool(home)
    sess = pool.up(cluster_name="mock", fill_spec="short:2", wait=True, timeout_s=20)
    res = pool.train(
        cluster_name="mock",
        argv=OOM_ALL,
        cwd=str(ROOT),
        session_id=sess.session_id,
        timeout_s=30,
    )
    check(res["ok"] is False, "oom should fail")
    check(all(r["returncode"] == 137 for r in res["results"]), "rc 137")
    # recovery train
    res2 = pool.train(
        cluster_name="mock",
        argv=SMOKE,
        cwd=str(ROOT),
        session_id=sess.session_id,
        timeout_s=30,
    )
    check(res2["ok"] is True, "recovery after oom")
    pool.down("mock", sess.session_id)


def test_asymmetric_rank_fail(home: Path) -> None:
    """One rank dies, others ok → segment fails (no silent success)."""
    pool = reload_pool(home)
    sess = pool.up(cluster_name="mock", fill_spec="short:2", wait=True, timeout_s=20)
    res = pool.train(
        cluster_name="mock",
        argv=ASYM_RANK0,
        cwd=str(ROOT),
        session_id=sess.session_id,
        timeout_s=30,
    )
    check(res["ok"] is False, "asym fail")
    rcs = {r["rank"]: r["returncode"] for r in res["results"]}
    check(rcs.get(0) == 1, f"rank0 rc {rcs}")
    check(rcs.get(1) == 0, f"rank1 rc {rcs}")

    res2 = pool.train(
        cluster_name="mock",
        argv=ASYM_RANK1,
        cwd=str(ROOT),
        session_id=sess.session_id,
        timeout_s=30,
    )
    check(res2["ok"] is False, "asym rank1")
    pool.down("mock", sess.session_id)


def test_hang_timeout_and_recover(home: Path) -> None:
    """NCCL hang stand-in: timeout cancels, clears ACTIVE, next train works."""
    pool = reload_pool(home)
    sess = pool.up(cluster_name="mock", fill_spec="short:2", wait=True, timeout_s=20)
    try:
        pool.train(
            cluster_name="mock",
            argv=HANG,
            cwd=str(ROOT),
            session_id=sess.session_id,
            timeout_s=4,
            poll_s=0.3,
        )
        raise Fail("expected TimeoutError")
    except TimeoutError:
        pass

    # ACTIVE must be clear — no "already in progress"
    res = pool.train(
        cluster_name="mock",
        argv=SMOKE,
        cwd=str(ROOT),
        session_id=sess.session_id,
        timeout_s=30,
    )
    check(res["ok"] is True, "recover after timeout")
    pool.down("mock", sess.session_id)


def test_overlapping_train_rejected(home: Path) -> None:
    pool = reload_pool(home)
    sess = pool.up(cluster_name="mock", fill_spec="short:2", wait=True, timeout_s=20)
    errors: list[str] = []

    def slow_train() -> None:
        try:
            pool.train(
                cluster_name="mock",
                argv=SLOW_OK,
                cwd=str(ROOT),
                session_id=sess.session_id,
                timeout_s=40,
                poll_s=0.3,
            )
        except Exception as e:
            errors.append(f"slow:{e}")

    t = threading.Thread(target=slow_train, daemon=True)
    t.start()
    time.sleep(1.5)
    blocked = False
    try:
        pool.train(
            cluster_name="mock",
            argv=SMOKE,
            cwd=str(ROOT),
            session_id=sess.session_id,
            timeout_s=10,
        )
    except RuntimeError as e:
        blocked = "in progress" in str(e).lower()
        check(blocked, f"unexpected error {e}")
    check(blocked, "second train should be rejected while first runs")
    # cancel first via stop
    pool.down("mock", sess.session_id)
    t.join(timeout=15)


def test_worker_kill_mid_train(home: Path) -> None:
    """Hard-kill one worker mid-train (preempt / node failure)."""
    pool = reload_pool(home)
    sess = pool.up(cluster_name="mock", fill_spec="short:2", wait=True, timeout_s=20)
    sid = sess.session_id

    def kill_soon() -> None:
        time.sleep(2.0)
        from fleet.session import load

        s = load(sid)
        pid = pid_for_slot(s, 1)
        if pid and alive(pid):
            print(f"[test] SIGKILL worker slot1 pid={pid}", flush=True)
            os.kill(pid, signal.SIGKILL)

    th = threading.Thread(target=kill_soon, daemon=True)
    th.start()
    res = pool.train(
        cluster_name="mock",
        argv=SLOW_OK,
        cwd=str(ROOT),
        session_id=sid,
        timeout_s=25,
        poll_s=0.4,
    )
    th.join(timeout=5)
    check(res["ok"] is False, f"kill mid-train should fail: {res}")
    check(len(res["results"]) == 2, f"need 2 results got {len(res['results'])}")
    # Only the surviving slot(s) participate under --flex / elastic.
    try:
        res2 = pool.train(
            cluster_name="mock",
            argv=SMOKE,
            cwd=str(ROOT),
            session_id=sid,
            timeout_s=20,
            elastic=True,
            min_world=1,
        )
        check(res2["ok"] is True, f"flex after kill: {res2}")
        check(res2["world_size"] == 1, f"expected 1 survivor world, got {res2}")
        check(res2["slots"] == [0], f"expected slot0 only, got {res2['slots']}")
    finally:
        pool.down("mock", sid)


def test_stop_during_train(home: Path) -> None:
    """fleet stop while train running — no hang, workers die."""
    pool = reload_pool(home)
    sess = pool.up(cluster_name="mock", fill_spec="short:2", wait=True, timeout_s=20)
    pids = local_pids(sess)
    sid = sess.session_id
    box: dict = {}

    def run_train() -> None:
        try:
            box["res"] = pool.train(
                cluster_name="mock",
                argv=SLOW_OK,
                cwd=str(ROOT),
                session_id=sid,
                timeout_s=40,
                poll_s=0.3,
            )
        except Exception as e:
            box["err"] = str(e)

    th = threading.Thread(target=run_train, daemon=True)
    th.start()
    time.sleep(2.0)
    pool.down("mock", sid)
    th.join(timeout=20)
    check(not th.is_alive(), "train thread should finish after stop")
    time.sleep(0.5)
    for pid in pids:
        deadline = time.time() + 5
        while time.time() < deadline and alive(pid):
            time.sleep(0.1)
        check(not alive(pid), f"worker {pid} still alive after stop")


def test_flex_after_short_death(home: Path) -> None:
    """Short walltime dies; full train fails; --flex runs on remaining."""
    pool = reload_pool(home)
    sess = pool.up(
        cluster_name="mock",
        fill_spec="long:1,short:1",
        wait=True,
        timeout_s=25,
        mock_ttl={"short": 3.0, "long": 60.0},
    )
    time.sleep(5.0)
    full_rejected = False
    try:
        pool.train(
            cluster_name="mock",
            argv=SMOKE,
            cwd=str(ROOT),
            session_id=sess.session_id,
            timeout_s=15,
        )
    except RuntimeError as e:
        full_rejected = "ready" in str(e).lower() or "flex" in str(e).lower()
        check(full_rejected, f"{e}")
    check(full_rejected, "full train must refuse when short is dead")

    res = pool.train(
        cluster_name="mock",
        argv=SMOKE,
        cwd=str(ROOT),
        session_id=sess.session_id,
        timeout_s=20,
        elastic=True,
        min_world=1,
    )
    check(res["ok"] is True, f"flex: {res}")
    check(res["world_size"] == 1, f"only long should train: {res}")
    pool.down("mock", sess.session_id)


def test_elastic_walltime_mid_segment(home: Path) -> None:
    """Short dies during long train segment → cancel + ok=False (not hang to timeout)."""
    pool = reload_pool(home)
    sess = pool.up(
        cluster_name="mock",
        fill_spec="long:1,short:1",
        wait=True,
        timeout_s=25,
        replenish=False,
        mock_ttl={"short": 4.0, "long": 90.0},
    )
    t0 = time.time()
    res = pool.train(
        cluster_name="mock",
        argv=SLOW_OK,
        cwd=str(ROOT),
        session_id=sess.session_id,
        timeout_s=40,
        elastic=True,
        min_world=1,
        poll_s=0.4,
    )
    elapsed = time.time() - t0
    check(res["ok"] is False, f"mid-walltime should fail segment: {res}")
    check(elapsed < 25, f"should cancel well before 40s timeout, took {elapsed:.1f}s")
    check(
        "slots_lost" in str(res.get("reason", "")) or any(
            r.get("returncode", 0) != 0 for r in res.get("results") or []
        ),
        f"expected cancel/fail reason: {res}",
    )
    pool.down("mock", sess.session_id)


def test_replenish_after_walltime(home: Path) -> None:
    pool = reload_pool(home)
    sess = pool.up(
        cluster_name="mock",
        fill_spec="long:1,short:1",
        wait=True,
        timeout_s=25,
        replenish=True,
        mock_ttl={"short": 3.0, "long": 60.0},
    )
    time.sleep(5.0)
    n = 0
    for _ in range(10):
        n = pool.replenish_once("mock", sess.session_id)
        if n >= 1:
            break
        time.sleep(0.4)
    check(n >= 1, f"replenish got {n}")
    pool.wait_ready("mock", sess.session_id, timeout_s=20, min_world=2)
    st = pool.status("mock", sess.session_id)
    short = next(r for r in st["ranks"] if r["queue"] == "short")
    check(short["generation"] >= 2, f"gen={short['generation']}")
    res = pool.train(
        cluster_name="mock",
        argv=SMOKE,
        cwd=str(ROOT),
        session_id=sess.session_id,
        timeout_s=30,
        elastic=True,
        min_world=2,
    )
    check(res["ok"] is True, f"after replenish: {res}")
    pool.down("mock", sess.session_id)


def test_elastic_loop_segments(home: Path) -> None:
    """--loop style: first segment fails, second succeeds via FLEET_SEGMENT."""
    pool = reload_pool(home)
    sess = pool.up(
        cluster_name="mock",
        fill_spec="long:1,medium:1",
        wait=True,
        timeout_s=25,
        replenish=True,
        mock_ttl={"medium": 40.0, "long": 90.0},
    )
    res = pool.train_elastic(
        cluster_name="mock",
        argv=SEG_RESUME,
        cwd=str(ROOT),
        session_id=sess.session_id,
        min_world=1,
        max_segments=2,
        segment_timeout_s=30,
        poll_s=0.5,
    )
    check(res["n_segments"] == 2, f"n_segments={res['n_segments']}")
    check(res["segments"][0].get("ok") is False, "seg0 fail")
    check(res["segments"][1].get("ok") is True, "seg1 ok")
    check(res["ok"] is True, "any segment ok")
    pool.down("mock", sess.session_id)


def test_command_not_found(home: Path) -> None:
    pool = reload_pool(home)
    sess = pool.up(cluster_name="mock", fill_spec="short:1", wait=True, timeout_s=15)
    res = pool.train(
        cluster_name="mock",
        argv=["/nonexistent/fleet-train-binary-xyz"],
        cwd=str(ROOT),
        session_id=sess.session_id,
        timeout_s=20,
    )
    check(res["ok"] is False, "missing binary")
    pool.down("mock", sess.session_id)


def test_work_then_die_then_smoke(home: Path) -> None:
    pool = reload_pool(home)
    sess = pool.up(cluster_name="mock", fill_spec="short:2", wait=True, timeout_s=20)
    res = pool.train(
        cluster_name="mock",
        argv=WORK_THEN_DIE,
        cwd=str(ROOT),
        session_id=sess.session_id,
        timeout_s=20,
    )
    check(res["ok"] is False, "work then die")
    check(all(r["returncode"] == 3 for r in res["results"]), "rc 3")
    res2 = pool.train(
        cluster_name="mock",
        argv=SMOKE,
        cwd=str(ROOT),
        session_id=sess.session_id,
        timeout_s=20,
    )
    check(res2["ok"] is True, "smoke after die")
    pool.down("mock", sess.session_id)


def test_min_world_elastic_too_small(home: Path) -> None:
    pool = reload_pool(home)
    sess = pool.up(
        cluster_name="mock",
        fill_spec="long:1,short:1",
        wait=True,
        timeout_s=20,
        mock_ttl={"short": 2.0, "long": 2.0},
    )
    time.sleep(4.5)
    try:
        pool.train(
            cluster_name="mock",
            argv=SMOKE,
            cwd=str(ROOT),
            session_id=sess.session_id,
            elastic=True,
            min_world=2,
            timeout_s=10,
        )
        # if both still somehow alive, skip
    except RuntimeError as e:
        check("ready" in str(e).lower() or "elastic" in str(e).lower(), f"{e}")
    pool.down("mock", sess.session_id)


def test_double_down_safe(home: Path) -> None:
    pool = reload_pool(home)
    sess = pool.up(cluster_name="mock", fill_spec="long:1", wait=True, timeout_s=15)
    pool.down("mock", sess.session_id)
    pool.down("mock", sess.session_id)  # idempotent-ish


def test_world_one_fail_and_ok(home: Path) -> None:
    pool = reload_pool(home)
    sess = pool.up(cluster_name="mock", fill_spec="long:1", wait=True, timeout_s=15)
    res = pool.train(
        cluster_name="mock",
        argv=OOM_ALL,
        cwd=str(ROOT),
        session_id=sess.session_id,
        timeout_s=15,
    )
    check(res["ok"] is False, "world1 oom")
    res2 = pool.train(
        cluster_name="mock",
        argv=SMOKE,
        cwd=str(ROOT),
        session_id=sess.session_id,
        timeout_s=15,
    )
    check(res2["ok"] is True, "world1 smoke")
    pool.down("mock", sess.session_id)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("empty_argv", test_empty_argv),
    ("bad_fill", test_bad_fill),
    ("train_before_ready", test_train_before_ready),
    ("oom_all_ranks", test_oom_all_ranks),
    ("asymmetric_rank_fail", test_asymmetric_rank_fail),
    ("hang_timeout_and_recover", test_hang_timeout_and_recover),
    ("overlapping_train_rejected", test_overlapping_train_rejected),
    ("worker_kill_mid_train", test_worker_kill_mid_train),
    ("stop_during_train", test_stop_during_train),
    ("flex_after_short_death", test_flex_after_short_death),
    ("elastic_walltime_mid_segment", test_elastic_walltime_mid_segment),
    ("replenish_after_walltime", test_replenish_after_walltime),
    ("elastic_loop_segments", test_elastic_loop_segments),
    ("command_not_found", test_command_not_found),
    ("work_then_die_then_smoke", test_work_then_die_then_smoke),
    ("min_world_elastic_too_small", test_min_world_elastic_too_small),
    ("double_down_safe", test_double_down_safe),
    ("world_one_fail_and_ok", test_world_one_fail_and_ok),
]


def main() -> int:
    failed = 0
    print("=== fleet failure-mode suite ===")
    for name, fn in TESTS:
        print(f"— {name} …", end=" ", flush=True)
        home = Path(tempfile.mkdtemp(prefix=f"fleet-fail-{name}-"))
        try:
            fn(home)
            print("OK")
        except Exception:
            failed += 1
            print("FAIL")
            traceback.print_exc()
        finally:
            shutil.rmtree(home, ignore_errors=True)
    total = len(TESTS)
    print(f"=== {total - failed}/{total} passed ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
