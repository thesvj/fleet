# Fleet — Feature Reference

Complete description of every feature in **fleet** (v0.4).  
Short operator guide: [README.md](README.md).

**Scope:** Built as a **practical hack for IIITD** (OMNI / Precision). Queue names, caps, tokens, and GRES match that site. Other clusters: extend `fleet/config.py` with your own `ClusterSpec` / `QueueSpec` entries — no core rewrite required.

---

## 1. What Fleet is

Fleet is a **multi-queue GPU pool** aimed at IIITD-style SLURM (OMNI, Precision) plus a local mock. Same code path can target another site if you register it in config.

**Core idea**

1. Take **one GPU per SLURM job** from flagged queues (`short` / `medium` / `long`).
2. Hold those jobs as a **pool of slots** with heartbeats on a shared filesystem.
3. Run your training script as **multi-job DDP** (ranks on different jobs/nodes).
4. Optionally **refill** short/medium when walltime kills them, while a long “anchor” lives.
5. Optionally train in **elastic segments**: stop a segment when GPUs die, re-rendezvous on remaining (or refilled) GPUs, resume from checkpoint.

It is **not** live NCCL rank-shrink mid-allreduce. When a peer dies, that train **segment** ends; the next segment is a new process group.

```
  login node (you)                    shared FS (~/.fleet or $FLEET_HOME)
  ┌──────────────────┐               ┌─────────────────────────────┐
  │ fleet start      │──sbatch──►    │ sessions/<id>/session.json  │
  │ fleet keep       │               │ ready/  meta/  commands/    │
  │ fleet run / stop │◄──squeue──    │ results/ rendezvous/ logs/  │
  └──────────────────┘               └──────────────▲──────────────┘
                                                    │ heartbeats + train cmd
         ┌──────────────┬──────────────┬────────────┴───┐
         │ job short    │ job medium   │ job long       │
         │ worker slot  │ worker slot  │ worker slot    │
         └──────────────┴──────────────┴────────────────┘
```

---

## 2. Design principles

| Principle | What it means in practice |
|---|---|
| **1 GPU = 1 job** | Matches IIITD QoS (often MaxJobsPA, not multi-GPU single job). |
| **Stable slot id** | Slot `0..world-1` never renumbers. Refill replaces the **job** in that seat (`generation++`). |
| **Dense DDP ranks per segment** | Training uses ranks `0..k-1` for the *currently participating* slots only. |
| **Shared FS is the bus** | Ready heartbeats, train commands, results, rendezvous master addr. |
| **Simulation first** | `-c mock` runs local processes; no SLURM, no tokens. |
| **Simple CLI names only** | `info check start ready show queue keep run stop` — no aliases. |
| **Token safety** | Real clusters need `--yes` (or interactive confirm). `check` never submits. |

---

## 3. Clusters and queues

Configured in `fleet/config.py` (verified notes as of mid-2026).

### 3.1 Cluster registry

| Name | Flag | Notes |
|---|---|---|
| **OMNI** | `-c omni` | Full GPU + MIG short; CUDA 12.8 modules (hosts/account in `config.py`) |
| **Precision** | `-c precision` | H100/A100/H200-class queues; CUDA 12.4 modules |
| **Mock** | `-c mock` | Local multi-process simulation; **0 tokens** |

Default cluster: env `FLEET_CLUSTER`, else **`mock`**.

### 3.2 Queue specs (per cluster)

Each queue has: `max_jobs`, `walltime`, `gres`, `cpus`, `mem`, `token_cost`, optional `nodelist` / `modules`.

#### OMNI (`-c omni`)

| Queue | Max jobs | Walltime | GRES | CPUs / mem | Token / job |
|---|---|---|---|---|---|
| short | 3 | 6h | `gpu:3g.71gb:1` (MIG) | 10 / 64G | **0.1** |
| medium | 2 | 1 day | `gpu:1` | 16 / 256G | **0.5** |
| long | 1 | 3 days | `gpu:1` | 20 / 512G | **1.0** |

Notes: short pinned to `dgxh200`; CUDA 12.8 only.

#### Precision (`-c precision`)

| Queue | Max jobs | Walltime | GRES | CPUs / mem | Token / job |
|---|---|---|---|---|---|
| short | 3 | 2h | `gpu:3g.40gb:1` (MIG) | 10 / 64G | **0.1** |
| medium | 2 | 12h | `gpu:1` | 16 / 256G | **0.5** |
| long | 1 | 3 days | `gpu:1` | 20 / 256G | **1.0** |

Notes: short on `gpu01`; CUDA 12.4.

#### Mock

Same max jobs (3/2/1), zero tokens, `mock:1` GRES, short walltimes for bookkeeping only. Real lifetime for sim is controlled by **`--fake-time`** (see §8).

### 3.3 Token model

```
est_tokens = Σ (token_cost[q] × count[q])   # per wave of concurrent jobs
```

- Tokens are charged when jobs **start** (even if idle holding the GPU).
- **Refill resubmits** short/medium → **more tokens** over multi-day runs.
- Full fill OMNI: `3×0.1 + 2×0.5 + 1×1.0 = 2.3` tokens per wave.

`fleet check` prints `est_tokens` without submitting.

---

## 4. Fill plan (`-f` / `--fill`)

Decides how many jobs per queue. Caps enforced against `max_jobs`.

| Spec | Meaning |
|---|---|
| `all` / `full` / `max` | Max of every queue (e.g. 1 long + 2 medium + 3 short) |
| `medium:2` | Exactly two medium |
| `long:1,medium:1,short:1` | One of each |
| `short` | All shorts (shorthand for that queue’s max) |

**Rank plan order** (stable): **long → medium → short**, so slot 0 is usually the longest-lived GPU (good for master / rank 0).

Errors if count &lt; 0 or count &gt; queue cap.

---

## 5. CLI — all commands

Entry: `python -m fleet` or `fleet` (with `PYTHONPATH` / install).  
Global: `--version`.

Shared options on most commands:

| Flag | Meaning |
|---|---|
| `-c` / `--cluster` | `omni` \| `precision` \| `mock` |
| `-s` / `--session` | Session id; default = latest active for that cluster |

### 5.1 `fleet info`

Show cluster name, login, notes, submit backend (`local+sbatch`), and each queue (max, gres, walltime, token).

```bash
fleet info -c omni
```

### 5.2 `fleet check`

**Dry preview only** — parse fill, world size, estimated tokens, planned ranks. JSON to stdout. **No submit.**

```bash
fleet check -c omni -f long:1,medium:1,short:1
```

### 5.3 `fleet start`

**Acquire GPUs** (submit one worker job per slot).

| Flag | Default | Meaning |
|---|---|---|
| `-f` / `--fill` | `all` | Which queues / counts |
| `--wait` / `--no-wait` | wait on | Block until ready (or timeout) |
| `--timeout` | 900 | Seconds for wait |
| `--dry-run` | off | Persist plan as dry-run ranks; no workers |
| `--local` | off | Force local processes (no SLURM) |
| `-y` / `--yes` | off | Skip confirm on real cluster |
| `--refill` | off | Enable short/medium auto-resubmit while main lives |
| `--main` | `long` | Anchor queue that must stay alive for refill |
| `--fake-time` | none | Mock only: per-queue self-exit TTL, e.g. `short:4,medium:10,long:40` |

**Safety on real clusters**

- Without `-y`: print plan; if TTY, ask “Submit billed jobs?”; if non-TTY, require `--yes`.
- With `--refill`, warns that short/medium will restart many times (more tokens).

Creates a session under `$FLEET_HOME/sessions/<id>/` and sets `active_<cluster>.session`.

### 5.4 `fleet ready`

Wait until enough slots have **fresh heartbeats**.

| Flag | Default | Meaning |
|---|---|---|
| `--timeout` | 900 | Max wait |
| `--min-gpu` | all slots | Ready when ≥ N GPUs are up |

If session has refill, may replenish dead non-anchor slots while waiting.

### 5.5 `fleet show`

Human table of slots:

- session / ready count / refill / main_alive / backend  
- per slot: **SLOT, QUEUE, GEN, STATE, SLURM compact/state, JOB, NODE, GPU**

Combines ready heartbeats + `squeue` (or local process view on mock).

### 5.6 `fleet queue`

SLURM-style view of **this session’s jobs**.

| Flag | Meaning |
|---|---|
| `-a` / `--all` | Also print full `squeue -u` for your user |

### 5.7 `fleet keep`

**Supervise loop** for multi-day refill (run in **tmux** on login node).

| Flag | Default | Meaning |
|---|---|---|
| `--every` | 10 | Poll interval (seconds) |
| `--timeout` | 0 | 0 = until main dies / SHUTDOWN / Ctrl-C |
| `--local` | off | Force local submit for replacements |

Requires session started with **`--refill`**. Stops when: SHUTDOWN (`fleet stop`), anchor dead, or timeout.

### 5.8 `fleet run`

Run training on pool workers. Everything after `--` is the train argv.

| Flag | Default | Meaning |
|---|---|---|
| `--cwd` | cwd | Working directory on workers |
| `--backend` | `auto` | `auto` \| `nccl` \| `gloo` |
| `--gloo` | off | Force gloo (CPU-friendly) |
| `--master-port` | 29500 | DDP master port |
| `--timeout` | 0 | Segment timeout (0 = none) |
| `--flex` | off | One segment on **currently ready** GPUs only |
| `--loop` | off | Elastic multi-segment until long dies |
| `--min-gpu` | 1 | Min ready GPUs for flex/loop |
| `--max-parts` | none | Cap number of segments in loop |
| `--part` | 0 | Segment index for a single (non-loop) run |

```bash
fleet run -c mock -- python examples/smoke_env.py
fleet run -c omni --flex --min-gpu 1 --backend gloo -- python train.py
fleet run -c omni --loop --min-gpu 1 -- python train.py --resume
```

### 5.9 `fleet stop`

Release everything for the session:

1. Write `SHUTDOWN`  
2. Clear active train  
3. Cancel in-process job handles  
4. `scancel` all historic SLURM job ids (or kill local PIDs) from `jobs.json`  
5. Mark ranks dead  

---

## 6. Session and shared filesystem

### 6.1 Home

| Path | Role |
|---|---|
| `$FLEET_HOME` | Override root (tests use `/tmp/...`) |
| default `~/.fleet` | Production home |
| `active_<cluster>.session` | Pointer to latest session id |
| `sessions/<id>/` | All state for one pool |

### 6.2 Session layout

```
sessions/<id>/
  session.json       # fill, ranks, replenish flags, mock_ttl
  jobs.json          # job ids (incl. history for scancel)
  SHUTDOWN           # present after stop (or request)
  ready/<slot>       # heartbeat timestamp
  meta/<slot>.json   # host, ip, gpu name, vram, pid
  commands/
    ACTIVE           # current train_id
    <train_id>.json  # argv, participants, segment, backend
    <train_id>.cancel
  results/<train_id>/<dense_rank>.json
  rendezvous/<train_id>/   # master_addr, barrier files
  logs/r<slot>_<queue>/    # worker stdout / sbatch scripts
```

### 6.3 Heartbeats

- Workers rewrite `ready/<slot>` every ~`FLEET_HEARTBEAT` seconds (default **2**).
- Stale if older than **90s** → not counted ready.
- **Job death wins over stale heartbeat**: if SLURM/local process is dead, slot is dead even if an old ready file remains.

### 6.4 Slot model (`RankSlot`)

| Field | Meaning |
|---|---|
| `rank` | Stable **slot id** 0..world-1 |
| `queue` | short / medium / long |
| `job_id` | Current SLURM or `local-<pid>` |
| `generation` | How many jobs launched in this seat (1, 2, …) |
| `job_history` | All job ids for this seat (for stop/scancel) |
| `state` | pending / ready / dead / dry-run |

---

## 7. Submit backends

`submit_worker` picks:

1. **Local** if `force_local` or cluster is `mock`  
2. Else **sbatch** shell script  

| Backend | Job id form | Cancel |
|---|---|---|
| local | `local-<pid>` | SIGTERM / kill process group |
| sbatch | numeric (parsable) | `scancel` |

No submitit path (kept out on purpose — less code, same cluster behavior).

**SLURM resources** from queue: partition, qos, account, cpus, mem, gres, time, optional nodelist, module loads, CUDA path, `PYTHONPATH` to fleet root, `B:USR1@180` signal.

---

## 8. Workers

Entrypoint: `fleet.worker.pool_worker` (sbatch `-c` / local subprocess).

### 8.1 Lifecycle

1. Detect hostname, IP, GPU name/VRAM (torch or nvidia-smi).  
2. Loop until SHUTDOWN or (mock) walltime TTL:  
   - Write ready heartbeat + meta  
   - If new `ACTIVE` train command → participate or skip  
3. Exit cleanly; **does not** write global SHUTDOWN on SIGTERM (so one short dying does not kill long).

### 8.2 Mock walltime (`--fake-time`)

Stored in `session.json` as `mock_ttl` (or env `FLEET_MOCK_TTL` JSON). Worker self-exits after N seconds for that queue — simulates SLURM TIMEOUT for refill/elastic tests.

```bash
fleet start -c mock -f long:1,medium:1,short:1 --refill \
  --fake-time short:4,medium:10,long:40 --wait
```

### 8.3 Train execution on worker

- **Slot id** vs **dense rank**: elastic maps `slot → dense_rank`; non-participants idle for that segment.  
- Shared-FS rendezvous (`setup_distributed`): rank 0 publishes `MASTER_ADDR`/`PORT`, file barrier, sets env only (**no** process group in the worker).  
- Train is a **subprocess** with full env; `fleet.init()` or your code owns torch PG.  
- Cancel if `.cancel` or SHUTDOWN; result written under dense rank id.

### 8.4 Environment exposed to your script

| Variable | Meaning |
|---|---|
| `RANK` | Dense DDP rank for this segment |
| `LOCAL_RANK` | Always `0` (one GPU per job) |
| `WORLD_SIZE` | Dense world size for this segment |
| `MASTER_ADDR` / `MASTER_PORT` | Rendezvous |
| `FLEET_BACKEND` | Backend string |
| `FLEET_TRAIN_ID` | Current train command id |
| `FLEET_SESSION` | Session id |
| `FLEET_SLOT` | Stable seat id (0..world-1 of the pool) |
| `FLEET_QUEUE` | `long` / `medium` / `short` |
| `FLEET_SEGMENT` | Segment index 0,1,2… (use for resume) |
| `FLEET_ELASTIC` | `1` if participant list set |
| `FLEET_MIN_WORLD` | Min world from orchestrator |

Also: `FLEET_HEARTBEAT`, `FLEET_RDZV_TIMEOUT` (worker/rendezvous tuning).

---

## 9. Refill (replenish)

### 9.1 When enabled

`fleet start ... --refill` sets `session.replenish = True` and `replenish_anchor` (default **`long`**, overridable with `--main`).

### 9.2 Behavior (`replenish_once`)

While **anchor** has ≥1 alive slot and no SHUTDOWN:

- For each **non-anchor** dead slot: clear ready, cancel old job, submit new job, `generation++`, append `job_history`.  
- **Never** auto-replaces the anchor queue itself.  
- If fill has no anchor queue, “any ready GPU” counts as anchor.

### 9.3 Who calls replenish

| Path | Behavior |
|---|---|
| `fleet keep` | Explicit loop (recommended for multi-day) |
| `fleet ready` / `wait_ready` | If replenish on, try once per poll |
| `fleet run` (single) | Once before train if replenish on |
| `fleet run --loop` | Before each segment |

### 9.4 Liveness detection

- **Local**: `Popen.poll()` if handle owned; else `/proc` — **zombies treated as dead** (important for mock refill).  
- **SLURM**: `squeue` state; terminal states (FAILED, TIMEOUT, CANCELLED, …) or missing from squeue → dead.  
- Ready heartbeat alone is **not** enough if the job process is gone.

---

## 10. Training modes

### 10.1 Full-world train (default `fleet run`)

Requires **all** slots ready. Dense ranks = slot ids 0..world-1.  
Fails if incomplete (suggests `--flex`).

### 10.2 Flex (`--flex`)

One segment on **currently ready** slots only. Maps to dense ranks 0..k-1.  
Requires `len(ready) >= --min-gpu`.  
If a participant dies mid-segment (elastic path), segment is **cancelled**.

### 10.3 Loop / elastic campaign (`--loop` → `train_elastic`)

```
while not SHUTDOWN and anchor alive and under max_parts:
  replenish_once (if enabled)
  wait_ready(min_world)
  train(elastic=True, segment=N)
  N += 1
```

- Your script **must checkpoint and resume** (`FLEET_SEGMENT`, your own ckpt path).  
- A segment ending (walltime / cancel / crash) is expected; next segment re-forms the world.  
- Without `--refill`, short/medium will not come back (warning printed).

### 10.4 Overlapping trains

If a previous train is still incomplete (results &lt; expected), new `run` **errors**. Stale finished ACTIVE is cleared.

### 10.5 Mid-run death policy

NCCL cannot drop ranks mid-allreduce safely. Fleet:

1. Detects missing ready for participants (with brief blip grace).  
2. Writes cancel file.  
3. Workers terminate train subprocess.  
4. Orchestrator finishes segment as failed/cancelled.  
5. Loop (if any) starts a **new** segment on remaining/refilled GPUs.

**Checkpoint often.**

---

## 11. Rendezvous (`fleet/rendezvous.py`)

| Feature | Detail |
|---|---|
| Master election | Dense rank 0 writes IP + port under `rendezvous/<train_id>/` |
| Barrier | Shared-FS file barrier for all dense ranks |
| Env only | Sets RANK/WORLD_SIZE/MASTER_*; no torch in worker parent |
| Train owns PG | Use `fleet.init()` or manual `init_process_group` in the script |

Backend auto (nccl/gloo) lives in `fleet.init()`, not rendezvous.

---

## 12. Train-script library (`import fleet`)

Optional helpers in `fleet/dist.py` (exported from package root):

| API | Role |
|---|---|
| `fleet.init()` | Read env ranks; signal handlers; optional `init_process_group` |
| `fleet.rank()` / `world_size()` | Cached identity |
| `fleet.step(opt, model)` | Allreduce grads (SUM / world) then optimizer step |
| `fleet.allreduce_grads(model)` | Manual allreduce |
| `fleet.checkpoint_if_needed()` | True after SIGTERM/SIGINT/SIGUSR1 |
| `fleet.barrier()` / `destroy()` | Sync / teardown |

World size 1 skips distributed init. Works with gloo on CPU for laptop smoke tests.

---

## 13. SLURM integration surface

| SLURM tool | Fleet use |
|---|---|
| `sbatch` | `start` / replenish |
| `squeue` | `show`, `queue`, `ready`, liveness |
| `scancel` | `stop` + job cancel |

Dead states recognized: BOOT_FAIL, CANCELLED, COMPLETED, DEADLINE, FAILED, NODE_FAIL, OUT_OF_MEMORY, PREEMPTED, TIMEOUT, SPECIAL_EXIT (+ compact codes).

`env_prefix` on OMNI injects correct `SLURM_CONF` and bin PATH for non-login shells.

---

## 14. Examples

| Script | Needs torch? | Purpose |
|---|---|---|
| `examples/smoke_env.py` | No | Check RANK/WORLD_SIZE/MASTER_* |
| `examples/smoke_ddp.py` | Yes | Allreduce sum correctness |
| `examples/train_toy.py` | Yes | Tiny train loop + `fleet.step` + checkpoint signal |

---

## 15. Simulation and tests

### 15.1 Laptop smoke

```bash
export PYTHONPATH=$PWD
export FLEET_HOME=/tmp/fleet-sim

fleet check  -c mock -f short:2
fleet start  -c mock -f short:2 --wait
fleet show   -c mock
fleet run    -c mock -- python examples/smoke_env.py
fleet stop   -c mock
```

### 15.2 Automated

```bash
make test
# or: python tests/run_sim.py
# optional: pytest (unit + e2e + squeue helpers)
```

Coverage intent:

- Fill caps, rank order, walltime parse, session ready TTL  
- e2e: start → show → run → run again → stop on mock  
- Failures: bad fill, train before ready, overlapping train  
- Cleanup: local workers terminated (no orphans)  
- Refill with fake walltime + zombie PID detection  

**Never** submit real SLURM from tests; always temp `FLEET_HOME`.

---

## 16. Multi-day operator recipe

Goal: hold **long** for ~3 days; churn medium/short for extra GPUs; train in segments.

```bash
# tmux pane 1 — acquire + enable refill
fleet check -c omni -f long:1,medium:1,short:1
fleet start -c omni -f long:1,medium:1,short:1 --refill -y --wait

# pane 2 — keep short/medium full
fleet keep -c omni --every 30

# pane 3 — elastic training (script must --resume)
fleet run -c omni --loop --min-gpu 1 --backend nccl -- \
  python train.py --ckpt ./ckpt --resume

# when finished
fleet stop -c omni
```

Rough capacity over 3 days (order of magnitude, queue-dependent):  
**1 long + several mediums + many shorts**, always with long as anchor when alive.

---

## 17. Safety checklist

1. Always develop/test on **`mock`** first.  
2. **`check`** before **`start`** (know token cost).  
3. Small fill first (`medium:2` or `short:2`), never `all` on first real day.  
4. Real submit: **`--yes`** or interactive confirm.  
5. Checkpoint in training code; use `FLEET_SEGMENT` for segment-aware resume.  
6. Always **`stop`** when done — idle billed GPUs burn tokens.  
7. Run `keep` / long `run --loop` inside **tmux** on the login node.  
8. Remember: refill multiplies token cost over days.

---

## 18. Package map

| Module | Responsibility |
|---|---|
| `cli.py` | argparse + command dispatch |
| `config.py` | Clusters, fill parse, tokens, walltime/mem |
| `schema.py` | QueueSpec, ClusterSpec, RankSlot, Session |
| `session.py` | Shared-FS state API |
| `executor.py` | Submit local / sbatch; squeue / scancel |
| `worker.py` | GPU holder + train subprocess |
| `rendezvous.py` | Multi-job rank join |
| `pool.py` | plan / up / ready / status / replenish / supervise / train / elastic / down |
| `dist.py` | Optional in-script helpers |

Target size: **≤ ~2k LOC** under `fleet/`. No HTTP controller, no dual worker stacks.

---

## 19. Environment variables (operator / test)

| Variable | Role |
|---|---|
| `FLEET_HOME` | State root |
| `FLEET_CLUSTER` | Default `-c` |
| `FLEET_HEARTBEAT` | Worker heartbeat interval (s) |
| `FLEET_RDZV_TIMEOUT` | Rendezvous wait (s) |
| `FLEET_MOCK_TTL` | JSON override for mock queue TTLs |
| `PYTHONPATH` | Must include fleet package root on login + workers |

---

## 20. Explicit non-features (today)

Not implemented (by design for v0.4):

- Multi-cluster single DDP world (OMNI + Precision + ADS in one process group)  
- Live NCCL shrink without segment restart  
- Auto-replace of the **long/anchor** job itself  
- HTTP / central broker service  
- Built-in training framework (you bring the train script)  
- Fair-share or cross-user scheduling  

---

## 21. Version

Package version: **0.4.1** (`fleet.__version__`).  
CLI: simple names only. Submit: local + sbatch. Rendezvous: env-only.
