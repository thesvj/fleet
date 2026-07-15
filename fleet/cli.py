"""
fleet CLI — simple commands:

  check   what will we take? (queues + cost)
  start   get GPUs
  ready   wait until GPUs are up
  show    see GPUs / jobs
  queue   SLURM-style job view
  keep    auto-refill short/medium while long runs
  run     train on the GPUs
  stop    free all GPUs
  info    cluster details
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from fleet import __version__
from fleet.config import get_cluster
from fleet.executor import backend_label
from fleet.pool import (
    down,
    plan,
    squeue_status,
    status,
    supervise,
    train,
    train_elastic,
    up,
    wait_ready,
)


def _cluster(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-c",
        "--cluster",
        default=os.environ.get("FLEET_CLUSTER", "mock"),
        help="omni | precision | mock",
    )
    p.add_argument("-s", "--session", default=None, help="session id (default: latest)")


def _fill(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-f",
        "--fill",
        default="all",
        help="which GPUs: long:1,medium:1,short:1  or  medium:2  or  all",
    )


def _parse_fake_time(raw: str | None) -> dict[str, float] | None:
    if not raw:
        return None
    out: dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        name, val = part.split(":", 1)
        out[name.strip()] = float(val)
    return out


def cmd_info(a: argparse.Namespace) -> int:
    c = get_cluster(a.cluster)
    print(f"cluster  {c.name}  ({c.user}@{c.login_host})")
    print(f"notes    {c.notes}")
    print(f"backend  {backend_label()}")
    print("queues")
    for n, q in c.queues.items():
        print(
            f"  {n:8}  max={q.max_jobs}  gres={q.gres}  "
            f"time={q.walltime}  token={q.token_cost}"
        )
    return 0


def cmd_check(a: argparse.Namespace) -> int:
    print(json.dumps(plan(a.cluster, a.fill), indent=2))
    return 0


def cmd_start(a: argparse.Namespace) -> int:
    if a.cluster != "mock" and not a.yes and not a.dry_run:
        p = plan(a.cluster, a.fill)
        print(
            json.dumps(
                {k: p[k] for k in ("cluster", "fill", "world_size", "est_tokens")},
                indent=2,
            )
        )
        if a.refill:
            print("refill=ON → short/medium will restart many times (more tokens) while long lives")
        if not sys.stdin.isatty():
            print("pass --yes to submit on a real cluster", file=sys.stderr)
            return 2
        if input("Submit billed jobs? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted")
            return 1

    sess = up(
        cluster_name=a.cluster,
        fill_spec=a.fill,
        wait=a.wait and not a.dry_run,
        timeout_s=a.timeout,
        force_local=a.local,
        session_id=a.session,
        dry_run=a.dry_run,
        replenish=a.refill,
        replenish_anchor=a.main,
        mock_ttl=_parse_fake_time(a.fake_time),
    )
    print(f"[fleet] session_id={sess.session_id}")
    return 0


def cmd_ready(a: argparse.Namespace) -> int:
    wait_ready(
        a.cluster,
        a.session,
        timeout_s=a.timeout,
        min_world=a.min_gpu,
    )
    return 0


def cmd_show(a: argparse.Namespace) -> int:
    st = status(a.cluster, a.session)
    print(
        f"session={st['session_id']}  cluster={st['cluster']}  "
        f"ready={st['ready']}/{st['world_size']}  "
        f"refill={st.get('replenish')}  main_alive={st.get('anchor_alive')}  "
        f"backend={st['backend']}"
    )
    print(f"dir={st['dir']}")
    print(
        f"{'SLOT':<5} {'QUEUE':<8} {'GEN':<4} {'STATE':<8} {'SLURM':<12} "
        f"{'JOB':<14} {'NODE':<14} GPU"
    )
    for r in st["ranks"]:
        print(
            f"{r['rank']:<5} {r['queue']:<8} {r.get('generation', 1):<4} {r['state']:<8} "
            f"{str(r.get('slurm_compact') or '-') + '/' + str(r.get('slurm_state') or '-'):<12} "
            f"{str(r.get('job_id') or '-'):<14} "
            f"{str(r.get('slurm_node') or '-'):<14} "
            f"{r.get('gpu') or '-'}"
        )
    return 0


def cmd_queue(a: argparse.Namespace) -> int:
    st = squeue_status(a.cluster, a.session, all_user=a.all)
    print(
        f"session={st['session_id']}  cluster={st['cluster']}  "
        f"ready={st['ready']}/{st['world_size']}  refill={st.get('replenish')}"
    )
    print("--- your fleet jobs ---")
    for line in st.get("squeue_session") or []:
        print(line)
    if a.all:
        print("--- all your cluster jobs (squeue -u) ---")
        print(st.get("squeue_user") or "")
    return 0


def cmd_keep(a: argparse.Namespace) -> int:
    supervise(
        a.cluster,
        a.session,
        poll_s=a.every,
        timeout_s=a.timeout,
        force_local=a.local,
    )
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    argv = list(a.argv)
    if not argv:
        print(
            "usage: fleet run -c CLUSTER [--loop] -- python train.py ...",
            file=sys.stderr,
        )
        return 2
    backend = "gloo" if a.gloo else a.backend
    if a.loop:
        res = train_elastic(
            cluster_name=a.cluster,
            argv=argv,
            cwd=a.cwd,
            backend=backend,
            master_port=a.master_port,
            session_id=a.session,
            min_world=a.min_gpu,
            max_segments=a.max_parts,
            segment_timeout_s=a.timeout,
        )
        print(json.dumps({"ok": res["ok"], "n_segments": res["n_segments"]}, indent=2))
        return 0 if res.get("ok") else 1

    res = train(
        cluster_name=a.cluster,
        argv=argv,
        cwd=a.cwd,
        backend=backend,
        master_port=a.master_port,
        session_id=a.session,
        timeout_s=a.timeout,
        elastic=a.flex,
        min_world=a.min_gpu,
        segment=a.part,
    )
    return 0 if res.get("ok") else 1


def cmd_stop(a: argparse.Namespace) -> int:
    down(a.cluster, a.session)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fleet",
        description="Get cluster GPUs and run multi-GPU training. Simple commands only.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  fleet check  -c mock -f long:1,medium:1,short:1
  fleet start  -c mock -f medium:2 --wait
  fleet show   -c mock
  fleet run    -c mock -- python examples/smoke_env.py
  fleet stop   -c mock

  # multi-day: refill short/medium while long runs
  fleet start  -c omni -f long:1,medium:1,short:1 --refill -y --wait
  fleet keep   -c omni --every 30
  fleet run    -c omni --loop --min-gpu 1 -- python train.py --resume
  fleet stop   -c omni
""",
    )
    p.add_argument("--version", action="version", version=f"fleet {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("info", help="show cluster queues")
    _cluster(s)
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("check", help="preview GPUs + token cost (no submit)")
    _cluster(s)
    _fill(s)
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("start", help="get GPUs from queues")
    _cluster(s)
    _fill(s)
    s.add_argument("--wait", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--timeout", type=float, default=900.0)
    s.add_argument("--dry-run", action="store_true", help="print plan only")
    s.add_argument("--local", action="store_true", help="local processes (no SLURM)")
    s.add_argument("-y", "--yes", action="store_true", help="no confirm on real cluster")
    s.add_argument(
        "--refill",
        action="store_true",
        help="when short/medium die, start a new one (while main/long lives)",
    )
    s.add_argument(
        "--main",
        default="long",
        help="main queue that must stay up for refill (default: long)",
    )
    s.add_argument(
        "--fake-time",
        default=None,
        help="mock only: fake walltime, e.g. short:4,medium:10,long:40",
    )
    s.set_defaults(func=cmd_start)

    s = sub.add_parser("ready", help="wait until GPUs are ready")
    _cluster(s)
    s.add_argument("--timeout", type=float, default=900.0)
    s.add_argument(
        "--min-gpu",
        type=int,
        default=None,
        help="ready when at least N GPUs are up (default: all)",
    )
    s.set_defaults(func=cmd_ready)

    s = sub.add_parser("show", help="show GPU slots and job state")
    _cluster(s)
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("queue", help="show jobs (like squeue)")
    _cluster(s)
    s.add_argument("--all", "-a", action="store_true", help="all your jobs on cluster")
    s.set_defaults(func=cmd_queue)

    s = sub.add_parser(
        "keep",
        help="keep refilling short/medium until long ends (use in tmux)",
    )
    _cluster(s)
    s.add_argument("--every", type=float, default=10.0, help="check every N seconds")
    s.add_argument("--timeout", type=float, default=0.0, help="0 = until main dies")
    s.add_argument("--local", action="store_true")
    s.set_defaults(func=cmd_keep)

    s = sub.add_parser("run", help="run training on the GPUs")
    _cluster(s)
    s.add_argument("--cwd", default=None)
    s.add_argument("--backend", default="auto", choices=["auto", "nccl", "gloo"])
    s.add_argument("--gloo", action="store_true", help="CPU-friendly backend")
    s.add_argument("--master-port", type=int, default=29500)
    s.add_argument("--timeout", type=float, default=0.0)
    s.add_argument(
        "--flex",
        action="store_true",
        help="use only GPUs that are up now (ok if short/medium died)",
    )
    s.add_argument(
        "--loop",
        action="store_true",
        help="refill + keep running parts until long dies (use checkpoint resume)",
    )
    s.add_argument("--min-gpu", type=int, default=1, help="min GPUs needed for --flex/--loop")
    s.add_argument("--max-parts", type=int, default=None, help="stop after N run parts")
    s.add_argument("--part", type=int, default=0, help="part number for single run")
    s.add_argument("argv", nargs=argparse.REMAINDER)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("stop", help="free all GPUs")
    _cluster(s)
    s.set_defaults(func=cmd_stop)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"[fleet] error: {exc}", file=sys.stderr)
        return 1
