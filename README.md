# Serverless-Function-Scheduling-with-Multi-Tenant-Fairness

Serverless Function Scheduling with Multi-Tenant Fairness

**COMP 6910 — Group 8**

## Problem Statement & Motivation

Serverless platforms (AWS Lambda, Azure Functions) enable on-demand function execution in multi-tenant environments. However, current schedulers (FIFO, round-robin) prioritize throughput / latency but fail to ensure proportional resource allocation across tenants. Specifically, they lack mechanisms to guarantee that each tenant receives a share of resources proportional to their fair entitlement. This unpredictability causes SLA violations and degrades service quality for smaller users—undermining serverless computing's core promise of predictable, on-demand execution.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

## Usage

### Single Experiment

```bash
python3 main.py --config configs/burst_test.yaml --scheduler fair_share
```

### Full Benchmark (all experiments x all schedulers)

```bash
python3 main.py --run-all
```

### Options

| Flag                 | Description                                       |
| -------------------- | ------------------------------------------------- |
| `--config <path>`    | Path to experiment YAML config                    |
| `--scheduler <name>` | fifo, round_robin, sjf, or fair_share             |
| `--run-all`          | Run all experiments and generate comparison plots |
| `--output <dir>`     | Output directory (default: results/)              |
| `--verbose`          | Log scheduling decisions                          |
| `--seed <int>`       | Override random seed                              |

## Output

Results are saved to `results/<experiment>/<scheduler>/`:

- `invocations.csv` — per-invocation data
- `metrics_summary.csv` — per-tenant metrics
- `experiment_result.json` — aggregated comparison

Comparison plots saved to `results/comparison/`.

## Running Tests

```bash
pytest tests/test_schedulers.py -v
```

## Schedulers

- **FIFO** — First-in-first-out, no fairness
- **Round-Robin** — Equal turns per tenant
- **SJF** — Shortest job first
- **Fair-Share** — Our proposed 2 phase scheduler

## License

MIT
