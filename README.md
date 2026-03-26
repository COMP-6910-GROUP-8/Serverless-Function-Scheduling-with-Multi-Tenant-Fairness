# Serverless-Function-Scheduling-with-Multi-Tenant-Fairness

Fairness-aware serverless function scheduling for multi-tenant cloud environments.

**COMP 6910 — Group 8**

## Problem

Serverless platforms promise all tenants the same SLA. In practice, large tenants' heavy or bursty workloads starve smaller tenants, causing effective SLA violations. This project simulates and evaluates a fairness-aware scheduler that prevents this.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

## Usage

### Single Experiment

```bash
python main.py --config configs/burst_test.yaml --scheduler fair_share
```

### Full Benchmark (all experiments x all schedulers)

```bash
python main.py --run-all
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
- **Fair-Share** — Our proposed scheduler using `Priority = α·FairShareDeficit + β·SLA_Urgency`

## License

MIT
