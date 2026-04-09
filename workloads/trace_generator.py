"""
/* USAGE:
  import numpy as np
  from simulator.config_loader import load_config
  from workloads.trace_generator import generate_tenants, generate_invocations

  config = load_config("configs/burst_test.yaml")
  rng = np.random.default_rng(config["simulation"]["random_seed"])

  tenants = generate_tenants(config, rng)
  invocations = generate_invocations(tenants, config, rng)

  print(len(tenants))       # 25
  print(len(invocations))   # thousands, sorted by arrival_time
*/
"""

import numpy as np
from simulator.models import Tenant, FunctionInvocation


def generate_tenants(config: dict, rng: np.random.Generator) -> list[Tenant]:
    """Create Tenant objects from config profiles with randomized arrival rates."""
    tenants = []
    for size, profile in config["tenants"].items():
        rate_low, rate_high = profile["arrival_rate"]
        for i in range(profile["count"]):
            tenant = Tenant(
                id=f"tenant_{size}_{i+1:02d}",
                size=size,
                arrival_rate=rng.uniform(rate_low, rate_high),
                function_profile=dict(profile["function_profile"]),
            )
            tenants.append(tenant)
    return tenants


def generate_invocations(
    tenants: list[Tenant], config: dict, rng: np.random.Generator
) -> list[FunctionInvocation]:
    """
    Generate invocation streams for all tenants using Poisson arrivals.
    Supports burst injection and per-tenant arrival rate overrides.
    """
    duration = config["experiment"]["duration"]
    archetypes = config["function_archetypes"]
    burst_cfg = config.get("experiment", {}).get("burst", None)
    overrides = config.get("experiment", {}).get("overrides", {})

    tenant_map = {t.id: t for t in tenants}
    for tid, ovr in overrides.items():
        if tid in tenant_map and "arrival_rate" in ovr:
            low, high = ovr["arrival_rate"]
            tenant_map[tid].arrival_rate = rng.uniform(low, high)

    all_invocations = []

    for tenant in tenants:
        func_types = list(tenant.function_profile.keys())
        func_weights = [tenant.function_profile[ft] for ft in func_types]
        total_w = sum(func_weights)
        func_probs = [w / total_w for w in func_weights]

        t = 0.0
        seq = 0

        while t < duration:
            rate = tenant.arrival_rate
            if (burst_cfg and tenant.id == burst_cfg["tenant_id"]
                    and burst_cfg["start_time"] <= t < burst_cfg["start_time"] + burst_cfg["duration"]):
                rate *= burst_cfg["multiplier"]

            inter_arrival = rng.exponential(1.0 / rate) if rate > 0 else duration
            t += inter_arrival
            if t >= duration:
                break

            seq += 1
            ftype = rng.choice(func_types, p=func_probs)
            arch = archetypes[ftype]
            dur_low, dur_high = arch["duration"]

            inv = FunctionInvocation(
                id=f"inv_{tenant.id}_{seq:05d}",
                tenant_id=tenant.id,
                function_type=ftype,
                cpu_demand=arch["cpu"],
                memory_demand=arch["memory"],
                base_duration=rng.uniform(dur_low, dur_high),
                arrival_time=t,
            )
            all_invocations.append(inv)

    all_invocations.sort(key=lambda inv: inv.arrival_time)
    return all_invocations
