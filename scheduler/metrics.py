"""
/* USAGE:
  from scheduler.metrics import (
      jains_fairness_index, compute_tenant_metrics, compute_experiment_summary,
      export_invocations_csv, export_metrics_csv, export_experiment_json,
  )

  tenant_metrics = compute_tenant_metrics(completed, tenants, config, duration=60)
  summary = compute_experiment_summary(tenant_metrics, scheduling_overheads)
  export_invocations_csv(completed, tenant_map, "results/steady_state/fifo/invocations.csv")
  export_metrics_csv(tenant_metrics, "results/steady_state/fifo/metrics_summary.csv")
*/
"""

import csv
import json
import os
import numpy as np
from simulator.models import FunctionInvocation, Tenant


def jains_fairness_index(fair_share_ratios: list[float]) -> float:
    """
    J(x) = (sum(xi))^2 / (n * sum(xi^2))
    Returns 1.0 for perfectly fair, lower for skewed.
    """
    if not fair_share_ratios:
        return 1.0
    n = len(fair_share_ratios)
    s = sum(fair_share_ratios)
    s2 = sum(x * x for x in fair_share_ratios)
    if s2 == 0:
        return 1.0
    return (s * s) / (n * s2)


def compute_tenant_metrics(
    invocations: list[FunctionInvocation],
    tenants: list[Tenant],
    config: dict,
    duration: float,
) -> list[dict]:
    """Compute per-tenant metrics from completed invocations."""
    sla_threshold = config["sla"]["p95_latency_threshold"]

    # Group invocations by tenant
    by_tenant: dict[str, list[FunctionInvocation]] = {}
    for inv in invocations:
        by_tenant.setdefault(inv.tenant_id, []).append(inv)

    # Total CPU-seconds across all tenants for fair share calculation
    total_cpu_seconds = sum(
        inv.cpu_demand * (inv.end_time - inv.start_time)
        for inv in invocations
        if inv.end_time is not None and inv.start_time is not None
    )
    active_tenants = [t for t in tenants if t.id in by_tenant]
    n_active = len(active_tenants)
    entitlement_per_tenant = total_cpu_seconds / n_active if n_active > 0 else 1.0

    results = []
    tenant_map = {t.id: t for t in tenants}

    for tenant in tenants:
        tenant_invs = by_tenant.get(tenant.id, [])
        if not tenant_invs:
            results.append({
                "tenant_id": tenant.id,
                "tenant_size": tenant.size,
                "avg_latency": 0.0,
                "p95_latency": 0.0,
                "throughput": 0.0,
                "sla_violation_rate": 0.0,
                "cold_start_rate": 0.0,
                "fair_share_ratio": 0.0,
            })
            continue

        latencies = [i.total_latency for i in tenant_invs if i.total_latency is not None]
        cold_starts = [i.cold_start for i in tenant_invs]
        n = len(tenant_invs)

        # CPU-seconds consumed by this tenant
        tenant_cpu_seconds = sum(
            inv.cpu_demand * (inv.end_time - inv.start_time)
            for inv in tenant_invs
            if inv.end_time is not None and inv.start_time is not None
        )

        results.append({
            "tenant_id": tenant.id,
            "tenant_size": tenant.size,
            "avg_latency": float(np.mean(latencies)) if latencies else 0.0,
            "p95_latency": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "throughput": n / duration if duration > 0 else 0.0,
            "sla_violation_rate": (
                sum(1 for l in latencies if l > sla_threshold) / len(latencies)
                if latencies else 0.0
            ),
            "cold_start_rate": sum(cold_starts) / n if n > 0 else 0.0,
            "fair_share_ratio": (
                tenant_cpu_seconds / entitlement_per_tenant
                if entitlement_per_tenant > 0 else 0.0
            ),
        })

    return results


def compute_experiment_summary(
    tenant_metrics: list[dict], scheduling_overheads: list[float]
) -> dict:
    """Aggregate tenant metrics into experiment-level summary."""
    # Jain's over SLA compliance scores: measures whether all tenants get
    # equitable service quality, not resource consumption equality
    sla_compliance = [1.0 - m["sla_violation_rate"] for m in tenant_metrics
                      if m["throughput"] > 0]  # only tenants with actual work
    violations = [m["sla_violation_rate"] for m in tenant_metrics]

    overhead_ms = [o * 1000 for o in scheduling_overheads] if scheduling_overheads else [0.0]

    return {
        "jains_fairness_index": jains_fairness_index(sla_compliance),
        "overall_sla_violation_rate": float(np.mean(violations)) if violations else 0.0,
        "avg_scheduling_overhead_ms": float(np.mean(overhead_ms)),
        "p95_scheduling_overhead_ms": float(np.percentile(overhead_ms, 95)),
    }


def export_invocations_csv(
    invocations: list[FunctionInvocation],
    tenants: dict[str, Tenant],
    output_path: str,
):
    """Write per-invocation CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "invocation_id", "tenant_id", "tenant_size", "function_type",
        "arrival_time", "start_time", "end_time", "server_id",
        "cold_start", "wait_time", "execution_time", "total_latency",
        "cpu_demand", "memory_demand",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for inv in invocations:
            execution_time = (
                (inv.end_time - inv.start_time)
                if inv.end_time is not None and inv.start_time is not None
                else None
            )
            writer.writerow({
                "invocation_id": inv.id,
                "tenant_id": inv.tenant_id,
                "tenant_size": tenants[inv.tenant_id].size if inv.tenant_id in tenants else "",
                "function_type": inv.function_type,
                "arrival_time": f"{inv.arrival_time:.6f}",
                "start_time": f"{inv.start_time:.6f}" if inv.start_time is not None else "",
                "end_time": f"{inv.end_time:.6f}" if inv.end_time is not None else "",
                "server_id": inv.server_id or "",
                "cold_start": inv.cold_start,
                "wait_time": f"{inv.wait_time:.6f}" if inv.wait_time is not None else "",
                "execution_time": f"{execution_time:.6f}" if execution_time is not None else "",
                "total_latency": f"{inv.total_latency:.6f}" if inv.total_latency is not None else "",
                "cpu_demand": inv.cpu_demand,
                "memory_demand": inv.memory_demand,
            })


def export_metrics_csv(tenant_metrics: list[dict], output_path: str):
    """Write per-tenant metrics summary CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "tenant_id", "tenant_size", "avg_latency", "p95_latency",
        "throughput", "sla_violation_rate", "cold_start_rate", "fair_share_ratio",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in tenant_metrics:
            writer.writerow(m)


def export_experiment_json(
    experiment_name: str, all_scheduler_results: dict[str, dict], output_path: str
):
    """Write aggregated experiment results JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {
        "experiment": experiment_name,
        "schedulers": all_scheduler_results,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
