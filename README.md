# Fleet

Get GPUs from IIITD queues and run multi-GPU training.

You can take **1 long + 2 medium + 3 short** at once (1 GPU per job). Fleet holds them and runs your train script across them.

> **Note:** This is a **hack tuned for IIITD clusters** (OMNI / Precision: short/medium/long QoS, token costs, MIG shorts). It is **not** a general multi-site orchestrator. Queue tables live in `fleet/config.py`.
>
> To use it elsewhere, edit `fleet/config.py`: add your cluster (account, partitions, gres, walltimes, caps, modules) the same way `omni` / `precision` / `mock` are defined, then pass `-c your_cluster`. The CLI and pool logic stay the same.

## Commands (simple names only)

| Command | What it does |
|---|---|
| `fleet info` | Show cluster queues |
| `fleet check` | Preview GPUs + token cost (**no submit**) |
| `fleet start` | **Get** GPUs |
| `fleet ready` | Wait until GPUs are up |
| `fleet show` | See slots / GPUs / job state |
| `fleet queue` | Jobs view (like `squeue`) |
| `fleet keep` | Auto-**refill** short/medium while long runs |
| `fleet run` | **Run** training |
| `fleet stop` | Free all GPUs |

## Install

```bash
cd path/to/fleet
export PYTHONPATH=$PWD
# optional: pip install -e ".[dev]"
```

## Quick test (laptop, free)

```bash
export PYTHONPATH=$PWD
export FLEET_HOME=/tmp/fleet-sim

fleet check  -c mock -f short:2
fleet start  -c mock -f short:2 --wait
fleet show   -c mock
fleet run    -c mock -- python examples/smoke_env.py
fleet stop   -c mock
```

Automated:

```bash
make test                  # sim + failure-mode suite
# or: python tests/run_sim.py
#     python tests/run_failures.py   # OOM, hang, preempt, flex, elastic…
```

## Real cluster (OMNI / Precision)

On **login node**, inside **tmux**:

```bash
fleet check -c omni -f medium:2
fleet start -c omni -f medium:2 -y --wait
fleet show  -c omni
fleet run   -c omni --backend gloo -- python examples/smoke_env.py
fleet stop  -c omni
```

| Cluster | Flag | Notes |
|---|---|---|
| OMNI | `-c omni` | IIITD OMNI queues (see `config.py`) |
| Precision | `-c precision` | IIITD Precision queues (see `config.py`) |
| Laptop test | `-c mock` | local multi-process, free |

Tokens are charged when jobs start (even if idle). **Do not commit** login hosts or lab accounts. Set them in your shell (or a private env file you never push):

```bash
export FLEET_ACCOUNT=...           # SLURM account
export FLEET_OMNI_HOST=...         # OMNI login host
export FLEET_PRECISION_HOST=...    # Precision login host
# optional: export FLEET_USER=...
```

## Which GPUs (`-f` / `--fill`)

```bash
-f medium:2                 # two medium
-f long:1,medium:1,short:1  # one of each
-f all                      # max: 1 long + 2 medium + 3 short
```

## Multi-day: long + refill short/medium

Short/medium die on walltime. Long can run 3 days. Fleet can:

1. **`--refill`** — when short/medium die, start a **new** one (while long is alive)
2. **`keep`** — loop that does refill for you
3. **`run --loop`** — keep training in parts until long ends (your script must **resume from checkpoint**)

```bash
# pane 1
fleet start -c omni -f long:1,medium:1,short:1 --refill -y --wait

# pane 2 — keep restarting short/medium
fleet keep -c omni --every 30

# pane 3 — train in parts until long ends
fleet run -c omni --loop --min-gpu 1 --backend nccl -- \
  python train.py --ckpt ./ckpt --resume

# when done
fleet stop -c omni
```

Over ~3 days you get roughly: **1 long + ~3 mediums + many shorts**, always with long.

| Flag | Meaning |
|---|---|
| `start --refill` | Restart short/medium when they die |
| `start --main long` | Main queue that must stay up (default long) |
| `keep` | Refill loop (tmux) |
| `run --flex` | This part uses only GPUs that are up now |
| `run --loop` | Many parts until long dies |
| `run --min-gpu 1` | OK to continue with only long |

Your train script sees:

| Env | Meaning |
|---|---|
| `RANK` / `WORLD_SIZE` | Ranks for **this part** |
| `FLEET_SEGMENT` | Part number 0,1,2… — use for resume |
| `FLEET_QUEUE` | long / medium / short |
| `FLEET_SLOT` | Stable seat id |

**Note:** If a GPU dies mid-allreduce, NCCL can hang. Fleet **stops that part** and starts a **new part** on remaining GPUs. Checkpoint often.

### Fake walltime (mock test)

```bash
fleet start -c mock -f long:1,medium:1,short:1 --refill \
  --fake-time short:4,medium:10,long:40 --wait
fleet keep -c mock --every 2 --timeout 25
```

## How it works

1. `start` → one SLURM job per GPU (`sbatch`)
2. Workers heart beat on shared disk
3. `show` / `queue` / `ready` use heartbeats + `squeue`
4. `run` starts train on all (or ready) GPUs
5. `stop` → `scancel` + kill workers

| SLURM | Used by |
|---|---|
| `sbatch` | `start` (and refill) |
| `squeue` | `show`, `queue`, `ready` |
| `scancel` | `stop` |

Submit path is **local (mock) + sbatch** only. Rendezvous sets env (`RANK`/`MASTER_*`); your train script (or `fleet.init()`) owns the process group.

## Training script

```python
import fleet
fleet.init()
# model on cuda:0 …
loss.backward()
fleet.step(opt, model)
if fleet.checkpoint_if_needed():
    save()
fleet.destroy()
```

Examples: `examples/smoke_env.py`, `smoke_ddp.py`, `train_toy.py`

## Safety

1. Always test with `mock` first  
2. `check` before `start` (know tokens)  
3. Small fill first (`medium:2`), not `all`  
4. Checkpoint in train code  
5. Always `stop` when done  

## Full feature reference

See **[FEATURES.md](FEATURES.md)** for queues, refill, elastic train, session layout, env vars, and safety details.

## Tests

```bash
make test
# or: python tests/run_sim.py
#     python tests/run_failures.py
```
