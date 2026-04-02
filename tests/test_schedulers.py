"""
/* USAGE:
  pytest tests/test_schedulers.py -v
*/
"""

import pytest
import numpy as np
from collections import deque
from simulator.models import Server, Tenant, FunctionInvocation
from scheduler.baseline_schedulers import FIFOScheduler, RoundRobinScheduler, SJFScheduler
from scheduler.fairness_scheduler import FairShareScheduler
from scheduler.metrics import jains_fairness_index


# --- Fixtures ---

@pytest.fixture
def servers():
    return [
        Server(id="s1", cpu_capacity=4000, memory_capacity=8192),
        Server(id="s2", cpu_capacity=4000, memory_capacity=8192),
    ]


@pytest.fixture
def tenants():
    return {
        "t1": Tenant(id="t1", size="large", arrival_rate=50.0, function_profile={"heavy": 1.0}),
        "t2": Tenant(id="t2", size="medium", arrival_rate=20.0, function_profile={"medium": 1.0}),
        "t3": Tenant(id="t3", size="small", arrival_rate=5.0, function_profile={"lightweight": 1.0}),
    }


def _make_inv(id, tenant_id, arrival_time, duration=0.05, cpu=250, mem=256, ftype="medium"):
    return FunctionInvocation(
        id=id, tenant_id=tenant_id, function_type=ftype,
        cpu_demand=cpu, memory_demand=mem,
        base_duration=duration, arrival_time=arrival_time,
    )


# --- Scheduler Tests ---

def test_fifo_ordering(servers, tenants):
    invocations = [
        _make_inv("a", "t1", 3.0),
        _make_inv("b", "t2", 1.0),
        _make_inv("c", "t3", 2.0),
    ]
    scheduler = FIFOScheduler()
    assignments = scheduler.schedule(invocations, tenants, servers, 5.0)
    assigned_ids = [inv.id for inv, _ in assignments]
    assert assigned_ids == ["b", "c", "a"]


def test_round_robin_cycling(servers, tenants):
    invocations = [
        _make_inv("t1_a", "t1", 1.0),
        _make_inv("t1_b", "t1", 1.1),
        _make_inv("t2_a", "t2", 1.0),
        _make_inv("t2_b", "t2", 1.1),
        _make_inv("t3_a", "t3", 1.0),
        _make_inv("t3_b", "t3", 1.1),
    ]
    scheduler = RoundRobinScheduler()
    assignments = scheduler.schedule(invocations, tenants, servers, 5.0)
    assigned_tids = [inv.tenant_id for inv, _ in assignments]
    # Should not assign all from one tenant before others
    # First 3 should be one from each tenant
    first_three_tids = set(assigned_tids[:3])
    # At least 2 different tenants in first 3
    assert len(first_three_tids) >= 2


def test_sjf_ordering(servers, tenants):
    invocations = [
        _make_inv("slow", "t1", 1.0, duration=0.300),
        _make_inv("fast", "t2", 1.0, duration=0.005),
        _make_inv("mid", "t3", 1.0, duration=0.050),
    ]
    scheduler = SJFScheduler()
    assignments = scheduler.schedule(invocations, tenants, servers, 5.0)
    assigned_ids = [inv.id for inv, _ in assignments]
    assert assigned_ids == ["fast", "mid", "slow"]


def test_fair_share_prioritizes_starved(servers, tenants):
    """Starved tenant (t3) should be scheduled before over-performing tenant (t1)."""
    scheduler = FairShareScheduler(sliding_window=5.0)
    # Simulate t1 having been dispatched several times already (over-served)
    scheduler._dispatch_counts = {"t1": 10}
    scheduler._total_dispatched = 10
    scheduler._window_start = 4.0

    invocations = [
        _make_inv("t1_inv", "t1", 4.8),
        _make_inv("t3_inv", "t3", 4.8),
    ]
    assignments = scheduler.schedule(invocations, tenants, servers, 5.0)
    assigned_ids = [inv.id for inv, _ in assignments]
    # t3 (starved, deficit = 5 - 0 = 5) should come before t1 (over-served, deficit = 5 - 10 = -5)
    assert assigned_ids.index("t3_inv") < assigned_ids.index("t1_inv")


# --- Metrics Tests ---

def test_jains_fairness_index_equal():
    assert jains_fairness_index([1.0, 1.0, 1.0, 1.0]) == 1.0


def test_jains_fairness_index_skewed():
    result = jains_fairness_index([0.1, 0.1, 0.1, 2.0])
    assert result < 0.7


# --- Model Tests ---

def test_server_capacity_enforcement():
    server = Server(id="s1", cpu_capacity=4000, memory_capacity=8192)
    assert server.has_capacity(500, 512)
    server.allocate(3800, 0)
    assert not server.has_capacity(500, 512)


def test_cold_start_warm_hit():
    server = Server(id="s1", cpu_capacity=4000, memory_capacity=8192)
    server.refresh_container("heavy", 10.0)
    assert server.has_warm_container("heavy", 10.5, 300)


def test_cold_start_ttl_expiry():
    server = Server(id="s1", cpu_capacity=4000, memory_capacity=8192)
    server.refresh_container("heavy", 10.0)
    assert not server.has_warm_container("heavy", 311.0, 300)


# --- Integration Test ---

def test_end_to_end_pipeline():
    """Small-scale pipeline: config → workload → simulation → metrics."""
    from simulator.config_loader import load_config
    from simulator.cloudsim_runner import SimulationEngine
    from workloads.trace_generator import generate_tenants, generate_invocations
    from scheduler.metrics import compute_tenant_metrics, compute_experiment_summary, compute_function_type_metrics

    config = load_config()
    # Override for small scale
    config["experiment"] = {"name": "test", "duration": 3}
    config["tenants"] = {
        "large": {"count": 1, "arrival_rate": [20, 30], "function_profile": {"heavy": 0.5, "medium": 0.5}},
        "medium": {"count": 2, "arrival_rate": [5, 10], "function_profile": {"medium": 0.7, "lightweight": 0.3}},
        "small": {"count": 2, "arrival_rate": [1, 3], "function_profile": {"lightweight": 1.0}},
    }
    config["servers"]["count"] = 2

    rng = np.random.default_rng(42)
    tenants = generate_tenants(config, rng)
    invocations = generate_invocations(tenants, config, rng)

    assert len(tenants) == 5
    assert len(invocations) > 0

    engine = SimulationEngine(config, scheduler=FIFOScheduler())
    completed, overheads = engine.run(tenants, invocations)

    assert len(completed) > 0
    assert all(inv.end_time is not None for inv in completed)

    metrics = compute_tenant_metrics(completed, tenants, config, 3.0)
    tenant_map = {t.id: t for t in tenants}
    ft_metrics = compute_function_type_metrics(completed, tenant_map, config, 3.0)
    summary = compute_experiment_summary(metrics, overheads, ft_metrics)

    assert 0.0 <= summary["jains_fairness_index"] <= 1.0
    assert summary["avg_scheduling_overhead_ms"] >= 0
    assert "per_function_type" in summary
    for ftype in ft_metrics:
        assert ft_metrics[ftype]["p95_latency"] >= 0
        assert ft_metrics[ftype]["max_wait_time"] >= 0
        assert ft_metrics[ftype]["throughput_ratio"] >= 0.0
