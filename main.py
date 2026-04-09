"""
/* USAGE:
  # Single experiment with one scheduler
  python main.py --config configs/burst_test.yaml --scheduler fair_share

  # Full benchmark (all experiments x all schedulers)
  python main.py --run-all

  # With options
  python main.py --run-all --output results/ --verbose --seed 123
*/
"""

import argparse
import os
import sys
import numpy as np

from simulator.config_loader import load_config
from simulator.cloudsim_runner import SimulationEngine
from workloads.trace_generator import generate_tenants, generate_invocations
from scheduler import FIFOScheduler, RoundRobinScheduler, SJFScheduler, FairShareScheduler
from scheduler.metrics import (
    compute_tenant_metrics, compute_experiment_summary,
    compute_function_type_metrics,
    export_invocations_csv, export_metrics_csv, export_experiment_json,
    generate_summary_md,
)
from plotting.plots import (
    plot_fairness_index,
    plot_p95_latency_by_size,
    plot_sla_violation_by_size, plot_sla_compliance_by_size,
    plot_throughput_equity, plot_throughput_boxplot,
    plot_ablation_heatmap, plot_scheduling_overhead, plot_p95_latency_by_function_type,
    plot_max_wait_by_function_type,
    plot_throughput_ratio_by_function_type,
    plot_stress_test_delta,
)

SCHEDULERS = ["fifo", "round_robin", "sjf", "fair_share"]
EXPERIMENTS = ["steady_state", "burst_test", "skewed_load", "stress_test"]


def get_scheduler(name: str, config: dict):
    """Map scheduler name to instance."""
    if name == "fifo":
        return FIFOScheduler()
    elif name == "round_robin":
        return RoundRobinScheduler()
    elif name == "sjf":
        return SJFScheduler()
    elif name == "fair_share":
        sched_cfg = config.get("scheduler", {})
        return FairShareScheduler(
            sliding_window=sched_cfg.get("sliding_window", 5.0),
            container_ttl=config["cold_start"]["container_ttl"],
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}. Choose from: {SCHEDULERS}")


def run_single(config: dict, scheduler_name: str, output_dir: str, verbose: bool) -> dict:
    """Run one experiment with one scheduler. Returns experiment summary dict."""
    exp_name = config["experiment"]["name"]
    duration = config["experiment"]["duration"]
    seed = config["simulation"]["random_seed"]
    rng = np.random.default_rng(seed)

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name} | Scheduler: {scheduler_name}")
    print(f"{'='*60}")

    tenants = generate_tenants(config, rng)
    invocations = generate_invocations(tenants, config, rng)
    print(f"Generated {len(invocations)} invocations for {len(tenants)} tenants ({duration}s)")

    scheduler = get_scheduler(scheduler_name, config)
    engine = SimulationEngine(config, scheduler=scheduler, verbose=verbose)
    completed, overheads = engine.run(tenants, invocations)
    print(f"Completed: {len(completed)} / {len(invocations)} invocations")

    tenant_map = {t.id: t for t in tenants}
    tenant_metrics = compute_tenant_metrics(completed, tenants, config, duration)
    ft_metrics = compute_function_type_metrics(completed, tenant_map, config, duration)
    summary = compute_experiment_summary(tenant_metrics, overheads, ft_metrics)

    out_path = os.path.join(output_dir, exp_name, scheduler_name)
    export_invocations_csv(completed, tenant_map, os.path.join(out_path, "invocations.csv"))
    export_metrics_csv(tenant_metrics, os.path.join(out_path, "metrics_summary.csv"))

    _print_summary(exp_name, scheduler_name, summary, tenant_metrics)

    return {
        "summary": summary,
        "tenant_metrics": tenant_metrics,
    }


def run_all(output_dir: str, verbose: bool, seed_override: int = None):
    """Run all experiments x all schedulers, generate comparison plots."""
    configs_dir = os.path.join(os.path.dirname(__file__), "configs")

    all_experiment_data = {}
    for exp_name in EXPERIMENTS:
        config_path = os.path.join(configs_dir, f"{exp_name}.yaml")
        experiment_data = {}

        for sched_name in SCHEDULERS:
            config = load_config(config_path, seed_override=seed_override)
            result = run_single(config, sched_name, output_dir, verbose)
            experiment_data[sched_name] = result

        all_experiment_data[exp_name] = experiment_data

        all_summaries = {s: d["summary"] for s, d in experiment_data.items()}
        json_path = os.path.join(output_dir, exp_name, "experiment_result.json")
        export_experiment_json(exp_name, all_summaries, json_path)

        comp_dir = os.path.join(output_dir, "comparison", exp_name)
        plot_fairness_index(experiment_data, os.path.join(comp_dir, "fairness_index.png"))
        plot_sla_violation_by_size(experiment_data, os.path.join(comp_dir, "sla_violation_by_size.png"))
        plot_sla_compliance_by_size(experiment_data, os.path.join(comp_dir, "sla_compliance_by_size.png"))
        plot_throughput_equity(experiment_data, os.path.join(comp_dir, "throughput_equity.png"))
        plot_p95_latency_by_size(experiment_data, os.path.join(comp_dir, "p95_latency_by_size.png"))
        plot_p95_latency_by_function_type(experiment_data, os.path.join(comp_dir, "p95_latency_by_function_type.png"))
        plot_max_wait_by_function_type(experiment_data, os.path.join(comp_dir, "max_wait_by_function_type.png"))
        plot_throughput_ratio_by_function_type(experiment_data, os.path.join(comp_dir, "throughput_ratio_by_function_type.png"))
        plot_throughput_boxplot(experiment_data, os.path.join(comp_dir, "throughput_boxplot.png"))

    if "steady_state" in all_experiment_data and "stress_test" in all_experiment_data:
        comp_dir = os.path.join(output_dir, "comparison")
        plot_stress_test_delta(
            all_experiment_data["steady_state"],
            all_experiment_data["stress_test"],
            os.path.join(comp_dir, "stress_test_delta.png"),
        )

    generate_summary_md(all_experiment_data, os.path.join(output_dir, "summary.md"))

    print(f"\n{'='*60}")
    print("Running scalability experiment...")
    print(f"{'='*60}")
    scalability_config = load_config(
        os.path.join(configs_dir, "scalability.yaml"), seed_override=seed_override
    )
    runs = scalability_config["experiment"]["runs"]
    scalability_data = {s: [] for s in SCHEDULERS}

    for run_cfg in runs:
        cfg = load_config(
            os.path.join(configs_dir, "scalability.yaml"), seed_override=seed_override
        )
        n_tenants = run_cfg["tenants"]
        n_servers = run_cfg["servers"]
        cfg["servers"]["count"] = n_servers

        total_default = sum(cfg["tenants"][s]["count"] for s in cfg["tenants"])
        for size in cfg["tenants"]:
            ratio = cfg["tenants"][size]["count"] / total_default
            cfg["tenants"][size]["count"] = max(1, round(n_tenants * ratio))

        for sched_name in SCHEDULERS:
            result = run_single(cfg, sched_name, output_dir, verbose)
            overhead = result["summary"]["avg_scheduling_overhead_ms"]
            scalability_data[sched_name].append((n_tenants, overhead))

    comp_dir = os.path.join(output_dir, "comparison")
    plot_scheduling_overhead(all_experiment_data, os.path.join(comp_dir, "scheduling_overhead.png"))

    print(f"\n{'='*60}")
    print("Running ablation experiment...")
    print(f"{'='*60}")
    ablation_config = load_config(
        os.path.join(configs_dir, "ablation.yaml"), seed_override=seed_override
    )
    sw_values = ablation_config["experiment"]["sliding_window_values"]
    ablation_data = {}

    for sw in sw_values:
        cfg = load_config(
            os.path.join(configs_dir, "ablation.yaml"), seed_override=seed_override
        )
        cfg.setdefault("scheduler", {})["sliding_window"] = sw

        result = run_single(cfg, "fair_share", output_dir, verbose)
        all_latencies = [m["avg_latency"] for m in result["tenant_metrics"] if m["avg_latency"] > 0]
        ablation_data[sw] = {
            "jains_fairness_index": result["summary"]["jains_fairness_index"],
            "avg_latency": float(np.mean(all_latencies)) if all_latencies else 0.0,
            "sla_violation_rate": result["summary"]["overall_sla_violation_rate"],
        }

    plot_ablation_heatmap(ablation_data, os.path.join(comp_dir, "ablation_heatmap.png"))

    print(f"\n{'='*60}")
    print(f"All experiments complete. Results in: {output_dir}")
    print(f"Comparison plots in: {os.path.join(output_dir, 'comparison')}")
    print(f"{'='*60}")


def _print_summary(exp_name: str, sched_name: str, summary: dict, tenant_metrics: list):
    """Print a formatted summary table to console."""
    size_latencies = {}
    for m in tenant_metrics:
        size = m["tenant_size"]
        if m["avg_latency"] > 0:
            size_latencies.setdefault(size, {"avg": [], "p95": []})
            size_latencies[size]["avg"].append(m["avg_latency"])
            size_latencies[size]["p95"].append(m["p95_latency"])

    print(f"\n  Jain's Fairness (resource): {summary['jains_fairness_index']:.4f}")
    print(f"  Jain's Fairness (SLA):      {summary['jains_sla_compliance']:.4f}")
    print(f"  SLA Violation Rate:         {summary['overall_sla_violation_rate']:.1%}")
    print(f"  Tenants Violating SLA:      {summary['tenant_sla_violation_rate']:.1%}")

    for size in ["small", "medium", "large"]:
        if size in size_latencies:
            avg = np.mean(size_latencies[size]["avg"])
            p95 = np.mean(size_latencies[size]["p95"])
            print(f"  Avg Latency ({size:>6s}):    {avg:.4f}s")
            print(f"  P95 Latency ({size:>6s}):    {p95:.4f}s")

    print(f"  Sched. Overhead (mean):   {summary['avg_scheduling_overhead_ms']:.3f}ms")
    print(f"  Sched. Overhead (P95):    {summary['p95_scheduling_overhead_ms']:.3f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="Fair-Share Scheduler Simulator for Multi-Tenant Serverless Platforms"
    )
    parser.add_argument("--config", type=str, help="Path to experiment YAML config")
    parser.add_argument("--scheduler", type=str, choices=SCHEDULERS, help="Scheduler to use")
    parser.add_argument("--run-all", action="store_true", help="Run all experiments and schedulers")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Log scheduling decisions")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    if not args.run_all and (not args.config or not args.scheduler):
        parser.error("Provide --config and --scheduler for a single run, or use --run-all")

    os.makedirs(args.output, exist_ok=True)

    if args.run_all:
        run_all(args.output, args.verbose, args.seed)
    else:
        config = load_config(args.config, seed_override=args.seed)
        run_single(config, args.scheduler, args.output, args.verbose)


if __name__ == "__main__":
    main()
