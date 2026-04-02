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


def plot_sla_violation_rate(experiment_data: dict, output_path: str):
    schedulers = _get_schedulers(experiment_data)
    values = [experiment_data[s]["summary"]["overall_sla_violation_rate"] for s in schedulers]
    colors = [SCHEDULER_COLORS[s] for s in schedulers]
    labels = [SCHEDULER_LABELS[s] for s in schedulers]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("SLA Violation Rate")
    ax.set_title("SLA Violation Rate by Scheduler")
    ax.set_ylim(0, max(max(values) * 1.2, 0.1) if values else 0.1)
    _save(output_path)


def plot_sla_violation_by_size(experiment_data: dict, output_path: str):
    """SLA violation rate by tenant size + Large:Small disparity ratio subplot."""
    schedulers = _get_schedulers(experiment_data)
    n_schedulers = len(schedulers)
    bar_width = 0.18

    # Collect per-scheduler, per-size violation rates
    sched_size_viol = {}
    for sched in schedulers:
        metrics = experiment_data[sched]["tenant_metrics"]
        size_viol = {}
        for size in SIZE_ORDER:
            vals = [m["sla_violation_rate"] for m in metrics if m["tenant_size"] == size]
            size_viol[size] = np.mean(vals) if vals else 0.0
        sched_size_viol[sched] = size_viol

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={"width_ratios": [2, 1]})

    # Left: grouped bar chart of violation rates by size
    x = np.arange(len(SIZE_ORDER))
    for idx, sched in enumerate(schedulers):
        values = [sched_size_viol[sched][s] for s in SIZE_ORDER]
        offset = (idx - n_schedulers / 2 + 0.5) * bar_width
        bars = ax1.bar(x + offset, values, bar_width,
                       label=SCHEDULER_LABELS[sched], color=SCHEDULER_COLORS[sched],
                       edgecolor="white", linewidth=0.5)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0.01:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                         f"{val:.1%}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax1.set_xlabel("Tenant Size")
    ax1.set_ylabel("SLA Violation Rate")
    ax1.set_title("SLA Violation Rate by Tenant Size")
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.capitalize() for s in SIZE_ORDER])
    ax1.legend(fontsize=8)

    # Right: Large:Small disparity ratio bar chart
    ratios = []
    ratio_labels = []
    ratio_colors = []
    for sched in schedulers:
        small_v = sched_size_viol[sched]["small"]
        large_v = sched_size_viol[sched]["large"]
        if small_v > 0.001:
            ratio = large_v / small_v
        elif large_v > 0.001:
            ratio = float("inf")
        else:
            ratio = 1.0  # both zero
        ratios.append(ratio)
        ratio_labels.append(SCHEDULER_LABELS[sched])
        ratio_colors.append(SCHEDULER_COLORS[sched])

    # Cap infinity for display, mark with special annotation
    display_ratios = []
    inf_indices = []
    finite_max = max((r for r in ratios if r != float("inf")), default=10.0)
    cap = max(finite_max * 1.3, 12.0)
    for i, r in enumerate(ratios):
        if r == float("inf"):
            display_ratios.append(cap)
            inf_indices.append(i)
        else:
            display_ratios.append(r)

    x2 = np.arange(len(schedulers))
    bars2 = ax2.bar(x2, display_ratios, color=ratio_colors, edgecolor="white", linewidth=0.5)

    # Add ratio labels on bars
    for i, (bar, r) in enumerate(zip(bars2, ratios)):
        label = "∞" if r == float("inf") else f"{r:.1f}×"
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Draw a reference line at 2.5x (our claimed threshold)
    ax2.axhline(y=2.5, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.text(len(schedulers) - 0.5, 2.7, "2.5× threshold", ha="right",
             fontsize=8, color="red", fontstyle="italic")
    # Draw a reference line at 1.0x (perfect equity)
    ax2.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.5)
    ax2.text(len(schedulers) - 0.5, 1.15, "1.0× (perfect equity)", ha="right",
             fontsize=7, color="gray")

    ax2.set_ylabel("Large:Small Violation Ratio")
    ax2.set_title("Disparity Ratio (Distribution of Pain)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(ratio_labels, fontsize=8, rotation=15)

    # Mark infinity bars with hatching
    for i in inf_indices:
        bars2[i].set_hatch("//")

    plt.tight_layout()
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


def plot_cold_start_by_size(experiment_data: dict, output_path: str):
    schedulers = _get_schedulers(experiment_data)
    n_schedulers = len(schedulers)
    bar_width = 0.18
    x = np.arange(len(SIZE_ORDER))

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, sched in enumerate(schedulers):
        metrics = experiment_data[sched]["tenant_metrics"]
        size_cs = {}
        for size in SIZE_ORDER:
            vals = [m["cold_start_rate"] for m in metrics if m["tenant_size"] == size]
            size_cs[size] = np.mean(vals) if vals else 0.0
        values = [size_cs[s] for s in SIZE_ORDER]
        offset = (idx - n_schedulers / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width,
               label=SCHEDULER_LABELS[sched], color=SCHEDULER_COLORS[sched],
               edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Tenant Size")
    ax.set_ylabel("Cold Start Rate")
    ax.set_title("Cold Start Rate by Tenant Size")
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
    alphas = sorted(ablation_data.keys())
    metric_names = ["jains_fairness_index", "avg_latency", "sla_violation_rate"]
    display_names = ["Fairness Index", "Avg Latency (s)", "SLA Violation Rate"]

    matrix = []
    for metric in metric_names:
        row = [ablation_data[a].get(metric, 0.0) for a in alphas]
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"\u03b1={a}" for a in alphas])
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names)
    ax.set_title("Ablation Study: \u03b1/\u03b2 Sensitivity")
    for i in range(len(display_names)):
        for j in range(len(alphas)):
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
