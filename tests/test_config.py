"""Unit tests for fill parsing and helpers."""

from __future__ import annotations

import pytest

from fleet.config import (
    get_cluster,
    mem_gb,
    parse_fill,
    rank_plan,
    token_cost,
    walltime_minutes,
)


def test_clusters_exist():
    for name in ("omni", "precision", "mock"):
        c = get_cluster(name)
        assert c.max_world() == 6
        assert set(c.queues) == {"short", "medium", "long"}


def test_unknown_cluster():
    with pytest.raises(KeyError):
        get_cluster("nope")


def test_fill_all():
    c = get_cluster("omni")
    fill = parse_fill("all", c)
    assert fill == {"short": 3, "medium": 2, "long": 1}
    assert abs(token_cost(c, fill) - 2.3) < 1e-9


def test_fill_partial():
    c = get_cluster("omni")
    fill = parse_fill("medium:2,short:1", c)
    assert fill == {"short": 1, "medium": 2, "long": 0}
    assert abs(token_cost(c, fill) - 1.1) < 1e-9


def test_fill_queue_name_only():
    c = get_cluster("mock")
    fill = parse_fill("long", c)
    assert fill["long"] == 1
    assert fill["short"] == 0


def test_fill_over_cap():
    c = get_cluster("omni")
    with pytest.raises(ValueError, match="cap"):
        parse_fill("long:2", c)


def test_fill_unknown_queue():
    c = get_cluster("omni")
    with pytest.raises(ValueError, match="queue"):
        parse_fill("gpu:1", c)


def test_rank_plan_order():
    ranks = rank_plan({"short": 2, "medium": 1, "long": 1})
    assert ranks == ["long", "medium", "short", "short"]


def test_walltime_minutes():
    assert walltime_minutes("6:00:00") == 360
    assert walltime_minutes("1-00:00:00") == 1440
    assert walltime_minutes("3-00:00:00") == 4320
    assert walltime_minutes("30") == 30


def test_mem_gb():
    assert mem_gb("64G") == 64
    assert mem_gb("256G") == 256
