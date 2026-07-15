"""Acquire GPUs → ready pool → replenish short/medium → elastic train → release."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from typing import Any

from fleet.config import get_cluster, parse_fill, rank_plan, token_cost
from fleet.executor import (
    Job,
    backend_label,
    is_dead_slurm_state,
    is_slurm_job_id,
    local_job_view,
    scancel_job,
    squeue_jobs,
    squeue_user,
    submit_worker,
)
from fleet.schema import RankSlot, Session
from fleet.session import (
    active_train,
    clear_shutdown,
    clear_slot_ready,
    clear_train,
    collect_results,
    load,
    new_id,
    ready_ranks,
    read_meta,
    request_cancel_train,
    request_shutdown,
    resolve_id,
    save,
    session_path,
    shutdown_requested,
    write_train,
)

# session_id → live job handles (same process that ran up/supervise)
_JOBS: dict[str, list[Job]] = {}


def plan(cluster_name: str, fill_spec: str | None = "all") -> dict[str, Any]:
    cluster = get_cluster(cluster_name)
    fill = parse_fill(fill_spec, cluster)
    queues = rank_plan(fill)
    return {
        "cluster": cluster.name,
        "fill": fill,
        "world_size": len(queues),
        "est_tokens": token_cost(cluster, fill),
        "backend": backend_label(),
        "notes": cluster.notes,
        "ranks": [{"rank": i, "queue": q} for i, q in enumerate(queues)],
    }


def _persist_jobs(sid: str, jobs: list[Job]) -> None:
    path = session_path(sid) / "jobs.json"
    prev: list[dict[str, Any]] = []
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except Exception:
            prev = []
    cur = [
        {"rank": j.rank, "queue": j.queue, "job_id": j.job_id, "backend": j.backend}
        for j in jobs
    ]
    seen = {(c["rank"], c["job_id"]) for c in cur}
    hist = [p for p in prev if (p.get("rank"), p.get("job_id")) not in seen]
    path.write_text(json.dumps(hist + cur, indent=2) + "\n")


def _enrich_slurm(cluster_name: str, ranks: list[RankSlot]) -> dict[str, dict[str, str]]:
    cluster = get_cluster(cluster_name)
    ids = [r.job_id for r in ranks if r.job_id]
    sq = squeue_jobs(cluster, [i for i in ids if i and is_slurm_job_id(i)])
    out: dict[str, dict[str, str]] = {}
    for jid in ids:
        if not jid:
            continue
        if jid in sq:
            out[jid] = sq[jid]
        else:
            view = local_job_view(jid)
            if is_slurm_job_id(jid):
                view = {
                    **view,
                    "state": "GONE",
                    "state_compact": "?",
                    "reason": "not-in-squeue (finished or unknown)",
                    "partition": "-",
                    "node": "-",
                }
            out[jid] = view
    return out


def slot_alive(
    cluster_name: str,
    slot: RankSlot,
    ready: set[int],
    slurm: dict[str, dict[str, str]],
) -> bool:
    """True if the slot job is still live. Job death wins over stale heartbeat."""
    info = slurm.get(slot.job_id or "", {})
    state = info.get("state", "")
    compact = info.get("state_compact", "")
    if (slot.job_id or "").startswith("local-"):
        for jlist in _JOBS.values():
            for j in jlist:
                if j.job_id == slot.job_id:
                    return j.is_running()
        return local_job_view(slot.job_id)["state"] == "RUNNING"
    if is_dead_slurm_state(state) or is_dead_slurm_state(compact) or state == "GONE":
        return False
    if slot.rank in ready:
        return True
    if compact in ("R", "PD", "CF") or state in ("RUNNING", "PENDING", "CONFIGURING"):
        return True
    return False


def _snapshot(cluster_name: str, sess: Session) -> tuple[set[int], dict[str, dict[str, str]]]:
    """Ready ranks + SLURM/local view for all slots (shared by ready/refill/status/train)."""
    ready = set(ready_ranks(sess.session_id, sess.world_size))
    slurm = _enrich_slurm(cluster_name, sess.ranks)
    return ready, slurm


def _maybe_replenish(
    cluster_name: str,
    sid: str,
    *,
    force_local: bool = False,
) -> Session:
    sess = load(sid)
    if sess.replenish:
        replenish_once(cluster_name, sid, force_local=force_local)
        sess = load(sid)
    return sess


def anchor_alive(cluster_name: str, sess: Session) -> bool:
    ready, slurm = _snapshot(cluster_name, sess)
    for r in sess.ranks:
        if r.queue == sess.replenish_anchor and slot_alive(cluster_name, r, ready, slurm):
            return True
    if not any(r.queue == sess.replenish_anchor for r in sess.ranks):
        return len(ready) > 0
    return False


def up(
    *,
    cluster_name: str,
    fill_spec: str | None = "all",
    wait: bool = True,
    timeout_s: float = 900.0,
    force_local: bool = False,
    session_id: str | None = None,
    dry_run: bool = False,
    replenish: bool = False,
    replenish_anchor: str = "long",
    mock_ttl: dict[str, float] | None = None,
) -> Session:
    cluster = get_cluster(cluster_name)
    fill = parse_fill(fill_spec, cluster)
    queues = rank_plan(fill)
    world = len(queues)
    if world < 1:
        raise ValueError("fill selects 0 GPUs; example: --fill long:1,medium:1,short:1")

    cost = token_cost(cluster, fill)
    print(f"[fleet] up cluster={cluster.name} fill={fill} world={world} tokens≈{cost}/wave")
    print(f"[fleet] backend={backend_label()} replenish={replenish} anchor={replenish_anchor}")
    if replenish:
        print(
            f"[fleet] replenish ON: re-submit short/medium when they die, "
            f"while ≥1 '{replenish_anchor}' slot is alive"
        )
    if cost > 0 and cluster.name != "mock":
        print("[fleet] warning: each GPU job is billed (incl. each replenish submit)")

    sid = session_id or new_id()
    sdir = session_path(sid)
    clear_shutdown(sid)
    clear_train(sid)

    sess = Session(
        session_id=sid,
        cluster=cluster.name,
        fill=fill,
        world_size=world,
        created_at=time.time(),
        ranks=[],
        replenish=replenish,
        replenish_anchor=replenish_anchor,
        mock_ttl=mock_ttl or {},
    )
    if dry_run:
        for i, q in enumerate(queues):
            sess.ranks.append(RankSlot(rank=i, queue=q, state="dry-run"))
        save(sess)
        print(f"[fleet] dry-run session={sid}")
        return sess

    save(sess)
    jobs: list[Job] = []
    for rank, qname in enumerate(queues):
        job = submit_worker(
            cluster=cluster,
            queue=cluster.queues[qname],
            rank=rank,
            world_size=world,
            session_dir=str(sdir),
            force_local=force_local or cluster.name == "mock",
        )
        jobs.append(job)
        sess.ranks.append(
            RankSlot(
                rank=rank,
                queue=qname,
                job_id=job.job_id,
                log_dir=job.log_dir,
                generation=1,
                job_history=[job.job_id],
            )
        )
        print(f"[fleet] slot {rank}/{world} queue={qname} job={job.job_id} gen=1")

    _persist_jobs(sid, jobs)
    save(sess)
    _JOBS[sid] = jobs
    print(f"[fleet] session={sid}\n[fleet] dir={sdir}")
    if wait:
        wait_ready(cluster.name, sid, timeout_s=timeout_s)
    return sess


def replenish_once(
    cluster_name: str,
    session_id: str | None = None,
    *,
    force_local: bool = False,
) -> int:
    """Re-submit dead non-anchor slots while anchor lives. Returns # new jobs."""
    sid = resolve_id(cluster_name, session_id)
    sess = load(sid)
    if not sess.replenish or shutdown_requested(sid) or not anchor_alive(cluster_name, sess):
        if sess.replenish and not shutdown_requested(sid) and not anchor_alive(cluster_name, sess):
            print("[fleet] replenish: anchor dead — stop refilling")
        return 0

    cluster = get_cluster(cluster_name)
    ready, slurm = _snapshot(cluster_name, sess)
    sdir = session_path(sid)
    jobs = list(_JOBS.get(sid, []))
    by_rank = {j.rank: j for j in jobs}
    submitted = 0

    for slot in sess.ranks:
        if slot.queue == sess.replenish_anchor:
            continue
        if slot_alive(cluster_name, slot, ready, slurm):
            continue

        clear_slot_ready(sid, slot.rank)
        old = by_rank.get(slot.rank)
        if old:
            try:
                old.cancel()
            except Exception:
                pass

        job = submit_worker(
            cluster=cluster,
            queue=cluster.queues[slot.queue],
            rank=slot.rank,
            world_size=sess.world_size,
            session_dir=str(sdir),
            force_local=force_local or cluster.name == "mock",
        )
        slot.generation += 1
        slot.job_id = job.job_id
        slot.log_dir = job.log_dir
        slot.state = "pending"
        slot.job_history.append(job.job_id)
        slot.hostname = None
        slot.ip = None
        by_rank[slot.rank] = job
        submitted += 1
        print(
            f"[fleet] replenish slot={slot.rank} queue={slot.queue} "
            f"gen={slot.generation} job={job.job_id}"
        )

    if submitted:
        new_jobs = [by_rank[r.rank] for r in sess.ranks if r.rank in by_rank]
        _JOBS[sid] = new_jobs
        _persist_jobs(sid, new_jobs)
        save(sess)
    return submitted


def supervise(
    cluster_name: str,
    session_id: str | None = None,
    *,
    poll_s: float = 10.0,
    timeout_s: float = 0.0,
    force_local: bool = False,
) -> None:
    """Loop: refill short/medium until anchor dies or SHUTDOWN / timeout."""
    sid = resolve_id(cluster_name, session_id)
    sess = load(sid)
    if not sess.replenish:
        raise RuntimeError("session has refill=false; re-start with --refill")
    print(
        f"[fleet] keep session={sid} main={sess.replenish_anchor} "
        f"every={poll_s}s (Ctrl-C or fleet stop to end)"
    )
    t0 = time.time()
    try:
        while True:
            if shutdown_requested(sid):
                print("[fleet] supervise: SHUTDOWN")
                return
            if timeout_s and time.time() - t0 > timeout_s:
                print("[fleet] supervise: timeout")
                return
            sess = load(sid)
            if not anchor_alive(cluster_name, sess):
                print("[fleet] supervise: anchor gone — done")
                return
            n = replenish_once(cluster_name, sid, force_local=force_local)
            st = status(cluster_name, sid)
            gens = {r["queue"]: r.get("generation", 1) for r in st["ranks"]}
            print(
                f"[fleet] supervise ready={st['ready']}/{st['world_size']} "
                f"replenished={n} gens={gens}"
            )
            time.sleep(poll_s)
    except KeyboardInterrupt:
        print("[fleet] supervise interrupted")


def wait_ready(
    cluster_name: str,
    session_id: str | None = None,
    timeout_s: float = 900.0,
    poll_s: float = 1.0,
    *,
    min_world: int | None = None,
) -> list[int]:
    """Wait until ≥ need slots are ready (need=world if min_world is None)."""
    sid = resolve_id(cluster_name, session_id)
    sess = load(sid)
    need = sess.world_size if min_world is None else min_world
    deadline = time.time() + timeout_s

    while True:
        sess = _maybe_replenish(cluster_name, sid)
        ready, slurm = _snapshot(cluster_name, sess)
        dead: list[str] = []
        pending: list[str] = []

        for r in sess.ranks:
            if r.rank in ready:
                continue
            info = slurm.get(r.job_id or "", {})
            state, compact = info.get("state", ""), info.get("state_compact", "")
            if is_dead_slurm_state(state) or is_dead_slurm_state(compact) or state == "GONE":
                if sess.replenish and r.queue != sess.replenish_anchor:
                    pending.append(f"rank{r.rank}/{r.queue} dead→replenish")
                else:
                    dead.append(
                        f"rank{r.rank} job={r.job_id} slurm={state}/{compact} "
                        f"reason={info.get('reason', '-')}"
                    )
            elif compact in ("PD", "CF") or state in ("PENDING", "CONFIGURING"):
                pending.append(f"rank{r.rank} job={r.job_id} ({state or compact})")
            elif compact == "R" or state == "RUNNING":
                pending.append(f"rank{r.rank} running wait_hb node={info.get('node', '?')}")

        print(
            f"[fleet] ready {len(ready)}/{sess.world_size} need>={need} {sorted(ready)}"
            + (f"  wait={pending}" if pending else "")
        )

        if dead and not sess.replenish:
            raise RuntimeError(
                "SLURM job(s) dead before worker ready:\n  "
                + "\n  ".join(dead)
                + f"\n  logs: {session_path(sid)}/logs/"
            )
        if dead and sess.replenish and not anchor_alive(cluster_name, sess):
            raise RuntimeError("anchor dead and slots failed:\n  " + "\n  ".join(dead))

        if len(ready) >= need:
            for r in sess.ranks:
                meta = read_meta(sid, r.rank)
                if meta:
                    r.state = "ready"
                    r.hostname = meta.get("hostname")
                    r.ip = meta.get("ip")
                    r.gpu_name = meta.get("gpu_name")
                    r.vram_mb = meta.get("vram_mb")
            save(sess)
            print(f"[fleet] ready ok ({len(ready)} slots)")
            return sorted(ready)

        if time.time() > deadline:
            raise TimeoutError(
                f"only {len(ready)}/{need} ready after {timeout_s}s "
                f"(logs: {session_path(sid)}/logs/)"
            )
        time.sleep(poll_s)


def status(cluster_name: str, session_id: str | None = None) -> dict[str, Any]:
    sid = resolve_id(cluster_name, session_id)
    sess = load(sid)
    ready, slurm = _snapshot(cluster_name, sess)
    rows = []
    for r in sess.ranks:
        meta = read_meta(sid, r.rank) or {}
        sq = slurm.get(r.job_id or "", {})
        rows.append(
            {
                "rank": r.rank,
                "queue": r.queue,
                "job_id": r.job_id,
                "state": "ready" if r.rank in ready else r.state,
                "generation": r.generation,
                "slurm_state": sq.get("state", "-"),
                "slurm_compact": sq.get("state_compact", "-"),
                "slurm_reason": sq.get("reason", "-"),
                "slurm_node": sq.get("node") or meta.get("hostname") or r.hostname or "-",
                "slurm_partition": sq.get("partition", r.queue),
                "slurm_elapsed": sq.get("elapsed", "-"),
                "hostname": meta.get("hostname") or r.hostname,
                "ip": meta.get("ip") or r.ip,
                "gpu": meta.get("gpu_name") or r.gpu_name,
                "vram_mb": meta.get("vram_mb") or r.vram_mb,
            }
        )
    return {
        "session_id": sid,
        "cluster": sess.cluster,
        "fill": sess.fill,
        "world_size": sess.world_size,
        "ready": len(ready),
        "replenish": sess.replenish,
        "replenish_anchor": sess.replenish_anchor,
        "anchor_alive": anchor_alive(cluster_name, sess),
        "ranks": rows,
        "dir": str(session_path(sid)),
        "backend": backend_label(),
    }


def squeue_status(
    cluster_name: str,
    session_id: str | None = None,
    *,
    all_user: bool = False,
) -> dict[str, Any]:
    cluster = get_cluster(cluster_name)
    st = status(cluster_name, session_id)
    lines = [
        (
            f"{r['job_id']:>14}  {r['slurm_compact']:>2}  {r['slurm_state']:<12}  "
            f"slot={r['rank']}  gen={r.get('generation', 1)}  part={r['slurm_partition']}  "
            f"node={r['slurm_node']}  reason={r['slurm_reason']}  fleet={r['state']}"
        )
        for r in st["ranks"]
    ]
    return {
        **st,
        "squeue_session": lines,
        "squeue_user": squeue_user(cluster) if all_user else None,
    }


def _participants(ready: list[int]) -> list[dict[str, int]]:
    return [{"slot": s, "dense_rank": i} for i, s in enumerate(sorted(ready))]


def _alive_in(
    cluster_name: str,
    sess: Session,
    slots: set[int],
) -> set[int]:
    """Participant slots whose SLURM/local job is still alive (not heartbeat-only)."""
    ready, slurm = _snapshot(cluster_name, sess)
    out: set[int] = set()
    for r in sess.ranks:
        if r.rank in slots and slot_alive(cluster_name, r, ready, slurm):
            out.add(r.rank)
    return out


def _finalize_train(
    sid: str,
    tid: str,
    *,
    expect: int,
    dense_keys: list[int],
    segment: int,
    slots: set[int],
    reason: str = "",
    missing_rc: int = 137,
) -> dict[str, Any]:
    """Collect results; fill missing dense ranks; clear ACTIVE. Never leaves a stuck train."""
    from fleet.session import write_result

    results = collect_results(sid, tid, expect, dense_ranks=dense_keys)
    have = {int(r["rank"]) for r in results}
    for d in dense_keys:
        if d not in have:
            write_result(
                sid,
                tid,
                d,
                missing_rc,
                "",
                f"fleet: no result from dense_rank={d} ({reason or 'lost/timeout'})",
            )
    results = collect_results(sid, tid, expect, dense_ranks=dense_keys)
    clear_train(sid)
    ok = bool(results) and all(r["returncode"] == 0 for r in results) and len(results) >= expect
    print(f"[fleet] train done ok={ok} seg={segment} reason={reason or 'complete'}")
    for r in sorted(results, key=lambda x: x["rank"]):
        print(f"  dense_rank={r['rank']} rc={r['returncode']}")
        if r["returncode"] != 0 and r.get("stderr_tail"):
            print(r["stderr_tail"][-800:])
    return {
        "ok": ok,
        "train_id": tid,
        "results": results,
        "segment": segment,
        "world_size": expect,
        "slots": sorted(slots),
        "reason": reason or "complete",
    }


def _reap_stale_train(sid: str, sess: Session) -> None:
    """If a previous ACTIVE train is unfinished but cancelled/shutdown, finalize it."""
    from fleet.session import cancel_train_requested, write_result

    prev = active_train(sid)
    if prev is None:
        return
    prev_n = int(prev.get("world_size") or len(prev.get("participants") or []) or sess.world_size)
    if prev_n < 1:
        prev_n = sess.world_size
    dense = list(range(prev_n))
    done = collect_results(sid, prev["train_id"], prev_n, dense_ranks=dense)
    if len(done) >= prev_n:
        clear_train(sid)
        return
    tid = prev["train_id"]
    if cancel_train_requested(sid, tid) or shutdown_requested(sid):
        have = {int(r["rank"]) for r in done}
        for d in dense:
            if d not in have:
                write_result(sid, tid, d, 137, "", "fleet: reaped stale cancelled train")
        clear_train(sid)
        return
    raise RuntimeError(f"train already in progress: {tid}")


def train(
    *,
    cluster_name: str,
    argv: list[str],
    cwd: str | None = None,
    backend: str = "auto",
    master_port: int = 29500,
    session_id: str | None = None,
    timeout_s: float = 0.0,
    poll_s: float = 0.5,
    elastic: bool = False,
    min_world: int = 1,
    segment: int = 0,
) -> dict[str, Any]:
    """
    One training segment on ready GPUs.
    elastic=True: dense ranks 0..k-1 on currently ready slots only.

    Failure modes handled:
    - rank exit ≠ 0 → ok=False, results collected
    - participant job death mid-run → cancel peers, fill missing, ok=False
    - timeout → cancel, finalize, raise TimeoutError after clearing ACTIVE
    - hang after cancel → fill missing ranks so next train can start
    """
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise ValueError("empty train command")

    sid = resolve_id(cluster_name, session_id)
    sess = _maybe_replenish(cluster_name, sid)
    # Heartbeat alone is not enough — dead jobs keep a ready file for ~90s.
    hb_ready = set(ready_ranks(sid, sess.world_size))
    alive = _alive_in(cluster_name, sess, set(range(sess.world_size)))
    usable = sorted(hb_ready & alive)

    if elastic:
        if len(usable) < min_world:
            raise RuntimeError(
                f"elastic train needs ≥{min_world} ready+alive, have {len(usable)} "
                f"(hb={sorted(hb_ready)} alive={sorted(alive)}); "
                f"fleet ready --min-gpu {min_world}"
            )
        participants = _participants(usable)
        expect = len(participants)
        slots = {p["slot"] for p in participants}
    else:
        if len(usable) < sess.world_size:
            raise RuntimeError(
                f"pool not fully ready ({len(usable)}/{sess.world_size} ready+alive, "
                f"hb={len(hb_ready)} alive={len(alive)}); "
                f"use --flex to run on remaining GPUs"
            )
        participants = None
        expect = sess.world_size
        slots = set(range(sess.world_size))

    _reap_stale_train(sid, sess)
    dense_keys = list(range(expect))
    tid = write_train(
        sid,
        argv=argv,
        cwd=str(Path(cwd or Path.cwd()).resolve()),
        backend=backend,
        master_port=master_port,
        participants=participants,
        segment=segment,
        world_size=expect,
        extra_env={
            "FLEET_ELASTIC": "1" if elastic else "0",
            "FLEET_SEGMENT": str(segment),
            "FLEET_MIN_WORLD": str(min_world),
        },
    )
    print(
        f"[fleet] train={tid} elastic={elastic} world={expect} "
        f"slots={sorted(slots)} seg={segment} argv={argv}"
    )

    deadline = time.time() + timeout_s if timeout_s else None
    cancel_after: float | None = None
    cancel_reason = ""

    try:
        while True:
            if shutdown_requested(sid):
                request_cancel_train(sid, tid)
                cancel_after = time.time() + 5.0
                cancel_reason = "shutdown"
                print("[fleet] train: SHUTDOWN — cancelling segment")

            # Job death wins over heartbeat (preempt / OOM-kill of worker / walltime).
            sess = load(sid)
            alive = _alive_in(cluster_name, sess, slots)
            missing_slots = slots - alive
            if missing_slots and cancel_after is None:
                time.sleep(1.5)
                sess = load(sid)
                alive = _alive_in(cluster_name, sess, slots)
                missing_slots = slots - alive
                if missing_slots:
                    print(f"[fleet] participant slots lost {sorted(missing_slots)} — cancel segment")
                    request_cancel_train(sid, tid)
                    cancel_after = time.time() + 8.0
                    cancel_reason = f"slots_lost={sorted(missing_slots)}"

            results = collect_results(sid, tid, expect, dense_ranks=dense_keys)
            if len(results) >= expect:
                return _finalize_train(
                    sid, tid, expect=expect, dense_keys=dense_keys,
                    segment=segment, slots=slots, reason="complete",
                )

            if cancel_after is not None and time.time() >= cancel_after:
                return _finalize_train(
                    sid, tid, expect=expect, dense_keys=dense_keys,
                    segment=segment, slots=slots, reason=cancel_reason or "cancelled",
                )

            if deadline and time.time() > deadline:
                print(f"[fleet] train timeout after {timeout_s}s — cancel {tid}")
                request_cancel_train(sid, tid)
                # brief grace for workers to report, then finalize + raise
                grace_end = time.time() + 6.0
                while time.time() < grace_end:
                    results = collect_results(sid, tid, expect, dense_ranks=dense_keys)
                    if len(results) >= expect:
                        break
                    time.sleep(0.3)
                out = _finalize_train(
                    sid, tid, expect=expect, dense_keys=dense_keys,
                    segment=segment, slots=slots, reason="timeout",
                )
                raise TimeoutError(
                    f"train {tid} incomplete after {timeout_s}s "
                    f"(got {len(out['results'])}/{expect} results)"
                )

            time.sleep(poll_s)
    except TimeoutError:
        raise
    except Exception:
        # Never leave ACTIVE stuck on unexpected errors.
        try:
            request_cancel_train(sid, tid)
            _finalize_train(
                sid, tid, expect=expect, dense_keys=dense_keys,
                segment=segment, slots=slots, reason="error",
            )
        except Exception:
            clear_train(sid)
        raise
    # unreachable
    return _finalize_train(
        sid, tid, expect=expect, dense_keys=dense_keys,
        segment=segment, slots=slots, reason="fallback",
    )


def train_elastic(
    *,
    cluster_name: str,
    argv: list[str],
    cwd: str | None = None,
    backend: str = "auto",
    master_port: int = 29500,
    session_id: str | None = None,
    min_world: int = 1,
    max_segments: int | None = None,
    segment_timeout_s: float = 0.0,
    poll_s: float = 2.0,
) -> dict[str, Any]:
    """
    Segments until SHUTDOWN / anchor dead / max_segments.
    Train script should resume from checkpoint (FLEET_SEGMENT).
    """
    sid = resolve_id(cluster_name, session_id)
    sess = load(sid)
    if not sess.replenish:
        print("[fleet] warning: loop without --refill will not restart short/medium")

    segments: list[dict[str, Any]] = []
    seg = 0
    while True:
        if shutdown_requested(sid):
            print("[fleet] elastic: SHUTDOWN")
            break
        sess = load(sid)
        if not anchor_alive(cluster_name, sess):
            print("[fleet] elastic: anchor dead — stop")
            break
        if max_segments is not None and seg >= max_segments:
            print(f"[fleet] elastic: max_segments={max_segments}")
            break

        _maybe_replenish(cluster_name, sid)
        try:
            wait_ready(
                cluster_name,
                sid,
                timeout_s=segment_timeout_s or 600,
                poll_s=poll_s,
                min_world=min_world,
            )
            res = train(
                cluster_name=cluster_name,
                argv=argv,
                cwd=cwd,
                backend=backend,
                master_port=master_port,
                session_id=sid,
                timeout_s=segment_timeout_s,
                elastic=True,
                min_world=min_world,
                segment=seg,
            )
        except Exception as exc:
            print(f"[fleet] elastic: segment {seg} error: {exc}")
            res = {"ok": False, "segment": seg, "error": str(exc)}
            if not anchor_alive(cluster_name, load(sid)):
                segments.append(res)
                break
            time.sleep(poll_s)
            segments.append(res)
            seg += 1
            continue

        segments.append(res)
        seg += 1
        time.sleep(1)

    return {"ok": any(s.get("ok") for s in segments), "segments": segments, "n_segments": len(segments)}


def down(cluster_name: str, session_id: str | None = None) -> None:
    sid = resolve_id(cluster_name, session_id)
    print(f"[fleet] down session={sid}")
    request_shutdown(sid)
    clear_train(sid)

    for job in _JOBS.pop(sid, []):
        try:
            job.cancel()
            print(f"[fleet] cancel slot={job.rank} job={job.job_id}")
        except Exception as exc:
            print(f"[fleet] cancel failed slot={job.rank}: {exc}")

    jobs_file = session_path(sid) / "jobs.json"
    if jobs_file.exists():
        cluster = get_cluster(cluster_name)
        for j in json.loads(jobs_file.read_text()):
            jid = str(j.get("job_id") or "")
            if jid.startswith("local-"):
                try:
                    os.kill(int(jid.split("-", 1)[1]), signal.SIGTERM)
                except (ProcessLookupError, ValueError):
                    pass
            elif is_slurm_job_id(jid):
                scancel_job(cluster, jid)
                print(f"[fleet] scancel {jid}")

    try:
        sess = load(sid)
        for r in sess.ranks:
            r.state = "dead"
        save(sess)
    except Exception:
        pass
    print("[fleet] down complete")
