"""
/* USAGE:
  from scheduler.metrics import (
      jains_fairness_index, compute_tenant_metrics, compute_experiment_summary,
      compute_function_type_metrics,
      export_invocations_csv, export_metrics_csv, export_experiment_json,
  )

  tenant_metrics = compute_tenant_metrics(completed, tenants, config, duration=60)
  ft_metrics = compute_function_type_metrics(completed, tenant_map, config, duration=60)
  summary = compute_experiment_summary(tenant_metrics, scheduling_overheads, ft_metrics)
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
    min_throughput_ratio = config["sla"]["min_throughput_ratio"]

    by_tenant: dict[str, list[FunctionInvocation]] = {}
    for inv in invocations:
        by_tenant.setdefault(inv.tenant_id, []).append(inv)

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
                "sla_violated": False,
                "invocations_violating_latency": 0,
                "total_invocations": 0,
                "cold_start_rate": 0.0,
                "fair_share_ratio": 0.0,
            })
            continue

        latencies = [i.total_latency for i in tenant_invs if i.total_latency is not None]
        cold_starts = [i.cold_start for i in tenant_invs]
        n = len(tenant_invs)

        tenant_cpu_seconds = sum(
            inv.cpu_demand * (inv.end_time - inv.start_time)
            for inv in tenant_invs
            if inv.end_time is not None and inv.start_time is not None
        )

        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
        invocations_over_threshold = sum(1 for lat in latencies if lat > sla_threshold)

        actual_throughput = n / duration if duration > 0 else 0.0
        expected_min_throughput = tenant.arrival_rate * min_throughput_ratio
        sla_violated = (p95 > sla_threshold) or (actual_throughput < expected_min_throughput)

        results.append({
            "tenant_id": tenant.id,
            "tenant_size": tenant.size,
            "arrival_rate": tenant.arrival_rate,
            "avg_latency": float(np.mean(latencies)) if latencies else 0.0,
            "p95_latency": p95,
            "throughput": actual_throughput,
            "throughput_ratio": (
                actual_throughput / tenant.arrival_rate
                if tenant.arrival_rate > 0 else 0.0
            ),
            "sla_violation_rate": (
                invocations_over_threshold / len(latencies)
                if latencies else 0.0
            ),
            "sla_violated": sla_violated,
            "invocations_violating_latency": invocations_over_threshold,
            "total_invocations": n,
            "cold_start_rate": sum(cold_starts) / n if n > 0 else 0.0,
            "fair_share_ratio": (
                tenant_cpu_seconds / entitlement_per_tenant
                if entitlement_per_tenant > 0 else 0.0
            ),
        })

    return results


def compute_function_type_metrics(
    invocations: list[FunctionInvocation],
    tenants: dict[str, Tenant],
    config: dict,
    duration: float,
) -> dict:
    """Compute per-function-type metrics: P95 latency, max wait time, throughput ratio."""
    by_ftype: dict[str, list[FunctionInvocation]] = {}
    for inv in invocations:
        by_ftype.setdefault(inv.function_type, []).append(inv)

    # Build expected invocation counts per function type from tenant profiles
    expected_per_ftype: dict[str, float] = {}
    for tenant in tenants.values():
        for ftype, weight in tenant.function_profile.items():
            expected_per_ftype[ftype] = expected_per_ftype.get(ftype, 0.0) + (
                tenant.arrival_rate * weight * duration
            )

    results = {}
    for ftype in ["lightweight", "medium", "heavy"]:
        ftype_invs = by_ftype.get(ftype, [])
        latencies = [i.total_latency for i in ftype_invs if i.total_latency is not None]
        wait_times = [i.wait_time for i in ftype_invs if i.wait_time is not None]

        expected_count = expected_per_ftype.get(ftype, 0.0)
        actual_count = len(ftype_invs)

        results[ftype] = {
            "p95_latency": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "max_wait_time": float(max(wait_times)) if wait_times else 0.0,
            "throughput_ratio": (
                actual_count / expected_count if expected_count > 0 else 0.0
            ),
            "total_completed": actual_count,
            "total_expected": round(expected_count),
        }

    return results


def compute_experiment_summary(
    tenant_metrics: list[dict],
    scheduling_overheads: list[float],
    function_type_metrics: dict | None = None,
) -> dict:
    """Aggregate tenant metrics into experiment-level summary."""
    fair_share_ratios = [m["fair_share_ratio"] for m in tenant_metrics
                         if m["throughput"] > 0]
    sla_compliance = [1.0 - m["sla_violation_rate"] for m in tenant_metrics
                      if m["throughput"] > 0]
    tenants_with_work = [m for m in tenant_metrics if m["throughput"] > 0]
    tenant_violation_count = sum(1 for m in tenants_with_work if m["sla_violated"])
    tenant_sla_violation_rate = (
        tenant_violation_count / len(tenants_with_work) if tenants_with_work else 0.0
    )

    total_violations = sum(m["invocations_violating_latency"] for m in tenant_metrics)
    total_invocations = sum(m["total_invocations"] for m in tenant_metrics)
    invocation_sla_violation_rate = (
        total_violations / total_invocations if total_invocations > 0 else 0.0
    )

    overhead_ms = [o * 1000 for o in scheduling_overheads] if scheduling_overheads else [0.0]

    per_size = {}
    for size in ["small", "medium", "large"]:
        size_metrics = [m for m in tenant_metrics if m["tenant_size"] == size and m["throughput"] > 0]
        if size_metrics:
            per_size[size] = {
                "avg_p95_latency": float(np.mean([m["p95_latency"] for m in size_metrics])),
                "avg_sla_violation_rate": float(np.mean([m["sla_violation_rate"] for m in size_metrics])),
                "avg_fair_share_ratio": float(np.mean([m["fair_share_ratio"] for m in size_metrics])),
                "avg_cold_start_rate": float(np.mean([m["cold_start_rate"] for m in size_metrics])),
                "tenant_count": len(size_metrics),
                "tenants_violating_sla": sum(1 for m in size_metrics if m["sla_violated"]),
            }

    summary = {
        "jains_fairness_index": jains_fairness_index(fair_share_ratios),
        "jains_sla_compliance": jains_fairness_index(sla_compliance),
        "overall_sla_violation_rate": invocation_sla_violation_rate,
        "tenant_sla_violation_rate": tenant_sla_violation_rate,
        "avg_scheduling_overhead_ms": float(np.mean(overhead_ms)),
        "p95_scheduling_overhead_ms": float(np.percentile(overhead_ms, 95)),
        "per_size": per_size,
    }

    if function_type_metrics is not None:
        summary["per_function_type"] = function_type_metrics

    return summary


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
        "tenant_id", "tenant_size", "arrival_rate", "avg_latency", "p95_latency",
        "throughput", "throughput_ratio", "sla_violation_rate", "sla_violated",
        "invocations_violating_latency", "total_invocations",
        "cold_start_rate", "fair_share_ratio",
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


def generate_summary_md(
    all_experiment_data: dict[str, dict[str, dict]],
    output_path: str,
):
    """Generate summary.md with scheduling overhead and distribution-of-pain tables.

    Args:
        all_experiment_data: {experiment_name: {scheduler_name: {"summary": ..., "tenant_metrics": ...}}}
        output_path: path to write summary.md
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    schedulers = ["fifo", "round_robin", "sjf", "fair_share"]
    sched_labels = {
        "fifo": "FIFO", "round_robin": "Round-Robin",
        "sjf": "SJF", "fair_share": "Fair-Share",
    }
    experiment_order = ["steady_state", "burst_test", "skewed_load", "stress_test"]
    experiments = [e for e in experiment_order if e in all_experiment_data]

    lines = ["# Experiment Results Summary\n"]

    # --- Scheduling Overhead ---
    lines.append("## Scheduling Overhead\n")
    for exp_name in experiments:
        exp_data = all_experiment_data[exp_name]
        lines.append(f"### {exp_name.replace('_', ' ').title()}\n")
        lines.append("| Scheduler | Avg Overhead (ms) | P95 Overhead (ms) |")
        lines.append("|-----------|------------------:|------------------:|")
        for s in schedulers:
            if s not in exp_data:
                continue
            summary = exp_data[s]["summary"]
            avg = summary["avg_scheduling_overhead_ms"]
            p95 = summary["p95_scheduling_overhead_ms"]
            lines.append(f"| {sched_labels[s]} | {avg:.3f} | {p95:.3f} |")
        lines.append("")

    # --- Distribution of Pain (SLA Violation by Tenant Size) ---
    lines.append("## Distribution of Pain — SLA Violation Rate by Tenant Size\n")
    for exp_name in experiments:
        exp_data = all_experiment_data[exp_name]
        lines.append(f"### {exp_name.replace('_', ' ').title()}\n")
        lines.append("| Scheduler | Small | Medium | Large |")
        lines.append("|-----------|------:|-------:|------:|")
        for s in schedulers:
            if s not in exp_data:
                continue
            metrics = exp_data[s]["tenant_metrics"]
            by_size = {}
            for size in ["small", "medium", "large"]:
                vals = [m["sla_violation_rate"] for m in metrics if m["tenant_size"] == size]
                by_size[size] = float(np.mean(vals)) if vals else 0.0

            lines.append(
                f"| {sched_labels[s]} "
                f"| {by_size['small']:.4f} "
                f"| {by_size['medium']:.4f} "
                f"| {by_size['large']:.4f} |"
            )
        lines.append("")

    # --- P95 Latency by Tenant Size ---
    lines.append("## P95 Latency by Tenant Size (seconds)\n")
    for exp_name in experiments:
        exp_data = all_experiment_data[exp_name]
        lines.append(f"### {exp_name.replace('_', ' ').title()}\n")
        lines.append("| Scheduler | Small | Medium | Large |")
        lines.append("|-----------|------:|-------:|------:|")
        for s in schedulers:
            if s not in exp_data:
                continue
            metrics = exp_data[s]["tenant_metrics"]
            by_size = {}
            for size in ["small", "medium", "large"]:
                vals = [m["p95_latency"] for m in metrics if m["tenant_size"] == size]
                by_size[size] = float(np.mean(vals)) if vals else 0.0
            lines.append(
                f"| {sched_labels[s]} "
                f"| {by_size['small']:.4f} "
                f"| {by_size['medium']:.4f} "
                f"| {by_size['large']:.4f} |"
            )
        lines.append("")

    # --- Per-Function-Type Metrics ---
    lines.append("## Per-Function-Type P95 Latency (seconds)\n")
    for exp_name in experiments:
        exp_data = all_experiment_data[exp_name]
        lines.append(f"### {exp_name.replace('_', ' ').title()}\n")
        lines.append("| Scheduler | Lightweight | Medium | Heavy |")
        lines.append("|-----------|------------:|-------:|------:|")
        for s in schedulers:
            if s not in exp_data:
                continue
            ft = exp_data[s]["summary"].get("per_function_type", {})
            lines.append(
                f"| {sched_labels[s]} "
                f"| {ft.get('lightweight', {}).get('p95_latency', 0):.4f} "
                f"| {ft.get('medium', {}).get('p95_latency', 0):.4f} "
                f"| {ft.get('heavy', {}).get('p95_latency', 0):.4f} |"
            )
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
