"""squeue helpers + status enrichment (mock / local)."""

from __future__ import annotations

from fleet.executor import is_dead_slurm_state, local_job_view, squeue_jobs
from fleet.config import get_cluster
from fleet.pool import squeue_status, status, up, down


def test_dead_states():
    assert is_dead_slurm_state("FAILED")
    assert is_dead_slurm_state("F")
    assert is_dead_slurm_state("CANCELLED+")
    assert not is_dead_slurm_state("RUNNING")
    assert not is_dead_slurm_state("R")
    assert not is_dead_slurm_state("PENDING")


def test_local_job_view_running_and_dead(fleet_home):
    import os
    import time

    # current process is alive
    view = local_job_view(f"local-{os.getpid()}")
    assert view["state"] == "RUNNING"
    assert view["state_compact"] == "R"

    # impossible pid
    view2 = local_job_view("local-99999999")
    assert view2["state"] == "COMPLETED"


def test_squeue_jobs_empty_for_mock():
    c = get_cluster("mock")
    assert squeue_jobs(c, ["12345"]) == {}


def test_status_includes_slurm_fields(fleet_home):
    sess = up(cluster_name="mock", fill_spec="short:1", wait=True, timeout_s=20)
    st = status("mock", sess.session_id)
    row = st["ranks"][0]
    assert "slurm_state" in row
    assert "slurm_compact" in row
    assert row["slurm_state"] in ("RUNNING", "COMPLETED")
    assert row["state"] == "ready"

    sq = squeue_status("mock", sess.session_id, all_user=False)
    assert len(sq["squeue_session"]) == 1
    assert "rank=0" in sq["squeue_session"][0]
    down("mock", sess.session_id)
