"""Session filesystem helpers."""

from __future__ import annotations

import time

from fleet.session import (
    active_train,
    clear_train,
    collect_results,
    ready_ranks,
    write_ready,
    write_result,
    write_train,
)
from fleet.schema import Session
from fleet.session import load, new_id, save


def test_session_roundtrip(fleet_home):
    sid = new_id()
    sess = Session(
        session_id=sid,
        cluster="mock",
        fill={"short": 2},
        world_size=2,
        created_at=time.time(),
    )
    save(sess)
    loaded = load(sid)
    assert loaded.session_id == sid
    assert loaded.world_size == 2


def test_ready_ttl(fleet_home):
    sid = new_id()
    write_ready(sid, 0, {"rank": 0})
    write_ready(sid, 1, {"rank": 1})
    assert ready_ranks(sid, 2, max_age_s=60) == [0, 1]
    # stale
    ready_path = fleet_home / "sessions" / sid / "ready" / "0"
    ready_path.write_text(f"{time.time() - 1000:.3f}\n")
    assert ready_ranks(sid, 2, max_age_s=30) == [1]


def test_train_command_and_results(fleet_home):
    sid = new_id()
    tid = write_train(
        sid,
        argv=["python", "-c", "print(1)"],
        cwd=".",
        backend="auto",
        master_port=29500,
    )
    cmd = active_train(sid)
    assert cmd is not None
    assert cmd["train_id"] == tid
    write_result(sid, tid, 0, 0, "ok", "")
    write_result(sid, tid, 1, 0, "ok", "")
    res = collect_results(sid, tid, 2)
    assert len(res) == 2
    clear_train(sid)
    assert active_train(sid) is None
