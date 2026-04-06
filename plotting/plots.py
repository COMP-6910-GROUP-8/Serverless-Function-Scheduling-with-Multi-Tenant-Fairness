"""
/* USAGE:
  from plotting.plots import plot_fairness_index, plot_p95_latency_by_size, ...

  plot_fairness_index(experiment_data, "results/comparison/fairness_index.png")
*/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file output
import matplotlib.pyplot as plt

SCHEDULER_COLORS = {
    "fifo": "#e74c3c",
    "round_robin": "#3498db",
    "sjf": "#f39c12",
    "fair_share": "#2ecc71",
}
SCHEDULER_LABELS = {
    "fifo": "FIFO",
    "round_robin": "Round-Robin",
    "sjf": "SJF",
    "fair_share": "Fair-Share (Ours)",
}
SIZE_ORDER = ["small", "medium", "large"]
FTYPE_ORDER = ["lightweight", "medium", "heavy"]


def _save(output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _get_schedulers(experiment_data: dict) -> list[str]:
    order = ["fifo", "round_robin", "sjf", "fair_share"]
    return [s for s in order if s in experiment_data]


def plot_fairness_index(experiment_data: dict, output_path: str):
    schedulers = _get_schedulers(experiment_data)
    values = [experiment_data[s]["summary"]["jains_fairness_index"] for s in schedulers]
    colors = [SCHEDULER_COLORS[s] for s in schedulers]
    labels = [SCHEDULER_LABELS[s] for s in schedulers]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="Perfect Fairness")
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_title("Jain's Fairness Index by Scheduler")
    ax.set_ylim(0, 1.1)
    ax.legend()
    _save(output_path)


def plot_p95_latency_by_size(experiment_data: dict, output_path: str, sla_threshold: float = 1.0):
    schedulers = _get_schedulers(experiment_data)
    n_schedulers = len(schedulers)
    bar_width = 0.18
    x = np.arange(len(SIZE_ORDER))

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, sched in enumerate(schedulers):
        metrics = experiment_data[sched]["tenant_metrics"]
        size_p95 = {}
        for size in SIZE_ORDER:
            vals = [m["p95_latency"] for m in metrics if m["tenant_size"] == size]
            size_p95[size] = np.mean(vals) if vals else 0.0
        values = [size_p95[s] for s in SIZE_ORDER]
        offset = (idx - n_schedulers / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width,
               label=SCHEDULER_LABELS[sched], color=SCHEDULER_COLORS[sched],
               edgecolor="white", linewidth=0.5)

    ax.axhline(y=sla_threshold, color="red", linestyle="--", linewidth=1, label=f"SLA ({sla_threshold}s)")
    ax.set_xlabel("Tenant Size")
    ax.set_ylabel("P95 Latency (seconds)")
    ax.set_title("P95 Latency by Tenant Size")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SIZE_ORDER])
    ax.legend(fontsize=8)
    _save(output_path)


def plot_sla_violation_by_size(experiment_data: dict, output_path: str):
    """SLA violation rate by tenant size — grouped bar chart."""
    schedulers = _get_schedulers(experiment_data)
    n_schedulers = len(schedulers)
    bar_width = 0.18
    x = np.arange(len(SIZE_ORDER))

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, sched in enumerate(schedulers):
        metrics = experiment_data[sched]["tenant_metrics"]
        size_viol = {}
        for size in SIZE_ORDER:
            vals = [m["sla_violation_rate"] for m in metrics if m["tenant_size"] == size]
            size_viol[size] = np.mean(vals) if vals else 0.0
        values = [size_viol[s] for s in SIZE_ORDER]
        offset = (idx - n_schedulers / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width,
                      label=SCHEDULER_LABELS[sched], color=SCHEDULER_COLORS[sched],
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.1%}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.set_xlabel("Tenant Size")
    ax.set_ylabel("SLA Violation Rate")
    ax.set_title("SLA Violation Rate by Tenant Size")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SIZE_ORDER])
    ax.legend(fontsize=8)
    _save(output_path)


def plot_sla_compliance_by_size(experiment_data: dict, output_path: str):
    """SLA compliance rate (1 - violation) by tenant size — positive framing of equitable service."""
    schedulers = _get_schedulers(experiment_data)
    n_schedulers = len(schedulers)
    bar_width = 0.18
    x = np.arange(len(SIZE_ORDER))

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, sched in enumerate(schedulers):
        metrics = experiment_data[sched]["tenant_metrics"]
        size_comp = {}
        for size in SIZE_ORDER:
            vals = [1.0 - m["sla_violation_rate"] for m in metrics if m["tenant_size"] == size]
            size_comp[size] = np.mean(vals) if vals else 1.0
        values = [size_comp[s] for s in SIZE_ORDER]
        offset = (idx - n_schedulers / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width,
                      label=SCHEDULER_LABELS[sched], color=SCHEDULER_COLORS[sched],
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.axhline(y=0.80, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(len(SIZE_ORDER) - 0.5, 0.81, "80% SLA minimum", ha="right",
            fontsize=8, color="red", fontstyle="italic")
    ax.set_xlabel("Tenant Size")
    ax.set_ylabel("SLA Compliance Rate")
    ax.set_title("SLA Compliance by Tenant Size")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SIZE_ORDER])
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8)
    _save(output_path)


def plot_throughput_equity(experiment_data: dict, output_path: str):
    """Throughput ratio (completed/expected) by tenant size — shows equitable service delivery."""
    schedulers = _get_schedulers(experiment_data)
    n_schedulers = len(schedulers)
    bar_width = 0.18
    x = np.arange(len(SIZE_ORDER))

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, sched in enumerate(schedulers):
        metrics = experiment_data[sched]["tenant_metrics"]
        size_ratio = {}
        for size in SIZE_ORDER:
            ratios = [m["throughput_ratio"] for m in metrics
                      if m["tenant_size"] == size and m.get("throughput_ratio", 0) > 0]
            size_ratio[size] = np.mean(ratios) if ratios else 0.0
        values = [size_ratio[s] for s in SIZE_ORDER]
        offset = (idx - n_schedulers / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width,
                      label=SCHEDULER_LABELS[sched], color=SCHEDULER_COLORS[sched],
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.axhline(y=0.80, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(len(SIZE_ORDER) - 0.5, 0.81, "80% SLA minimum", ha="right",
            fontsize=8, color="red", fontstyle="italic")
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.5)
    ax.text(len(SIZE_ORDER) - 0.5, 1.01, "100% (all invocations served)", ha="right",
            fontsize=7, color="gray")
    ax.set_xlabel("Tenant Size")
    ax.set_ylabel("Throughput Ratio (Completed / Expected)")
    ax.set_title("Throughput Equity by Tenant Size")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SIZE_ORDER])
    ax.legend(fontsize=8)
    _save(output_path)


def plot_throughput_boxplot(experiment_data: dict, output_path: str):
    schedulers = _get_schedulers(experiment_data)
    data = []
    labels = []
    for sched in schedulers:
        metrics = experiment_data[sched]["tenant_metrics"]
        throughputs = [m["throughput"] for m in metrics if m["throughput"] > 0]
        data.append(throughputs)
        labels.append(SCHEDULER_LABELS[sched])

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, sched in zip(bp["boxes"], schedulers):
        patch.set_facecolor(SCHEDULER_COLORS[sched])
        patch.set_alpha(0.7)
    ax.set_ylabel("Throughput (inv/sec)")
    ax.set_title("Throughput Distribution by Scheduler")
    _save(output_path)


def plot_ablation_heatmap(ablation_data: dict, output_path: str):
    params = sorted(ablation_data.keys())
    metric_names = ["jains_fairness_index", "avg_latency", "sla_violation_rate"]
    display_names = ["Fairness Index", "Avg Latency (s)", "SLA Violation Rate"]

    matrix = []
    for metric in metric_names:
        row = [ablation_data[p].get(metric, 0.0) for p in params]
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels([f"sw={p}s" for p in params])
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names)
    ax.set_title("Ablation Study: Sliding Window Sensitivity")
    for i in range(len(display_names)):
        for j in range(len(params)):
            ax.text(j, i, f"{matrix[i][j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    _save(output_path)


def plot_scheduling_overhead(all_experiment_data: dict, output_path: str):
    """Grouped bar chart of avg scheduling overhead across experiments for all schedulers."""
    experiments = list(all_experiment_data.keys())
    schedulers = list(SCHEDULER_LABELS.keys())
    n_schedulers = len(schedulers)
    bar_width = 0.18
    x = np.arange(len(experiments))

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, sched in enumerate(schedulers):
        values = []
        for exp in experiments:
            exp_data = all_experiment_data.get(exp, {})
            overhead = exp_data.get(sched, {}).get("summary", {}).get("avg_scheduling_overhead_ms", 0.0)
            values.append(overhead)
        offset = (idx - n_schedulers / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width,
                      label=SCHEDULER_LABELS[sched], color=SCHEDULER_COLORS[sched],
                      edgecolor="white", linewidth=0.5)
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    exp_labels = [e.replace("_", " ").title() for e in experiments]
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Avg Scheduling Overhead (ms)")
    ax.set_title("Scheduling Overhead Comparison Across Experiments")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _save(output_path)


def plot_p95_latency_by_function_type(experiment_data: dict, output_path: str, sla_threshold: float = 1.0):
    schedulers = _get_schedulers(experiment_data)
    n_schedulers = len(schedulers)
    bar_width = 0.18
    x = np.arange(len(FTYPE_ORDER))

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, sched in enumerate(schedulers):
        ft = experiment_data[sched]["summary"].get("per_function_type", {})
        values = [ft.get(f, {}).get("p95_latency", 0.0) for f in FTYPE_ORDER]
        offset = (idx - n_schedulers / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width,
               label=SCHEDULER_LABELS[sched], color=SCHEDULER_COLORS[sched],
               edgecolor="white", linewidth=0.5)

    ax.axhline(y=sla_threshold, color="red", linestyle="--", linewidth=1, label=f"SLA ({sla_threshold}s)")
    ax.set_xlabel("Function Type")
    ax.set_ylabel("P95 Latency (seconds)")
    ax.set_title("P95 Latency by Function Type")
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in FTYPE_ORDER])
    ax.legend(fontsize=8)
    _save(output_path)


def plot_max_wait_by_function_type(experiment_data: dict, output_path: str):
    schedulers = _get_schedulers(experiment_data)
    n_schedulers = len(schedulers)
    bar_width = 0.18
    x = np.arange(len(FTYPE_ORDER))

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, sched in enumerate(schedulers):
        ft = experiment_data[sched]["summary"].get("per_function_type", {})
        values = [ft.get(f, {}).get("max_wait_time", 0.0) for f in FTYPE_ORDER]
        offset = (idx - n_schedulers / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width,
               label=SCHEDULER_LABELS[sched], color=SCHEDULER_COLORS[sched],
               edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Function Type")
    ax.set_ylabel("Max Wait Time (seconds)")
    ax.set_title("Max Wait Time by Function Type")
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in FTYPE_ORDER])
    ax.legend(fontsize=8)
    _save(output_path)


def plot_stress_test_delta(
    steady_data: dict, stress_data: dict, output_path: str
):
    """
    Bar chart showing the change in small-tenant SLA violation rate from
    steady_state to stress_test for each scheduler. Measures robustness
    to workload composition changes.
    """
    schedulers = _get_schedulers(steady_data)

    steady_small = {}
    stress_small = {}
    for sched in schedulers:
        s_metrics = steady_data[sched]["tenant_metrics"]
        t_metrics = stress_data[sched]["tenant_metrics"]
        s_vals = [m["sla_violation_rate"] for m in s_metrics if m["tenant_size"] == "small"]
        t_vals = [m["sla_violation_rate"] for m in t_metrics if m["tenant_size"] == "small"]
        steady_small[sched] = np.mean(s_vals) if s_vals else 0.0
        stress_small[sched] = np.mean(t_vals) if t_vals else 0.0

    labels = [SCHEDULER_LABELS[s] for s in schedulers]
    colors = [SCHEDULER_COLORS[s] for s in schedulers]
    steady_vals = [steady_small[s] for s in schedulers]
    stress_vals = [stress_small[s] for s in schedulers]
    deltas = [stress_vals[i] - steady_vals[i] for i in range(len(schedulers))]

    x = np.arange(len(schedulers))
    bar_width = 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: side-by-side steady vs stress small-tenant violation rates
    ax1.bar(x - bar_width / 2, steady_vals, bar_width, label="Steady State",
            color=[c + "88" for c in colors], edgecolor="white")
    ax1.bar(x + bar_width / 2, stress_vals, bar_width, label="Stress Test",
            color=colors, edgecolor="white")
    for i in range(len(schedulers)):
        for val, offset in [(steady_vals[i], -bar_width / 2), (stress_vals[i], bar_width / 2)]:
            if val > 0.0005:
                ax1.text(x[i] + offset, val + 0.001, f"{val:.2%}",
                         ha="center", va="bottom", fontsize=8)
            else:
                ax1.text(x[i] + offset, 0.002, "0%", ha="center", va="bottom", fontsize=8)
    ax1.set_xlabel("Scheduler")
    ax1.set_ylabel("Small Tenant SLA Violation Rate")
    ax1.set_title("Small Tenant Violations: Steady vs Stress")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()

    # Right: delta (stress - steady)
    bar_colors = ["#e74c3c" if d > 0.001 else "#2ecc71" for d in deltas]
    bars = ax2.bar(x, deltas, color=bar_colors, edgecolor="white")
    for bar, d in zip(bars, deltas):
        label = f"+{d:.2%}" if d > 0 else f"{d:.2%}"
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.001 if d >= 0 else bar.get_height() - 0.003,
                 label, ha="center", va="bottom" if d >= 0 else "top", fontsize=9,
                 fontweight="bold")
    ax2.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax2.set_xlabel("Scheduler")
    ax2.set_ylabel("Delta (Stress - Steady)")
    ax2.set_title("Workload Robustness: Change in Small Tenant Violations")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)

    plt.tight_layout()
    _save(output_path)


def plot_throughput_ratio_by_function_type(experiment_data: dict, output_path: str):
    schedulers = _get_schedulers(experiment_data)
    n_schedulers = len(schedulers)
    bar_width = 0.18
    x = np.arange(len(FTYPE_ORDER))

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, sched in enumerate(schedulers):
        ft = experiment_data[sched]["summary"].get("per_function_type", {})
        values = [ft.get(f, {}).get("throughput_ratio", 0.0) for f in FTYPE_ORDER]
        offset = (idx - n_schedulers / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width,
               label=SCHEDULER_LABELS[sched], color=SCHEDULER_COLORS[sched],
               edgecolor="white", linewidth=0.5)

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="100% Throughput")
    ax.set_xlabel("Function Type")
    ax.set_ylabel("Throughput Ratio (completed / expected)")
    ax.set_title("Throughput Ratio by Function Type")
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in FTYPE_ORDER])
    ax.legend(fontsize=8)
    _save(output_path)
