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


def plot_p95_latency_by_size(experiment_data: dict, output_path: str, sla_threshold: float = 0.1):
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


def plot_scalability(scalability_data: dict, output_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    for sched, points in scalability_data.items():
        if not points:
            continue
        points.sort(key=lambda p: p[0])
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        ax.plot(x, y, marker="o",
                label=SCHEDULER_LABELS.get(sched, sched),
                color=SCHEDULER_COLORS.get(sched, "gray"))

    ax.set_xlabel("Number of Tenants")
    ax.set_ylabel("Mean Scheduling Overhead (ms)")
    ax.set_title("Scheduling Overhead vs. Tenant Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(output_path)
