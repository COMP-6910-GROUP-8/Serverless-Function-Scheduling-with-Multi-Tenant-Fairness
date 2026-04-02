"""
/* USAGE:
  from scheduler.fairness_scheduler import FairShareScheduler

  scheduler = FairShareScheduler(
      alpha=0.6, beta=0.4, sliding_window=1.0,
      sla_latency_threshold=1.000, sla_min_throughput_ratio=0.80,
  )
  assignments = scheduler.schedule(pending_invocations, tenants_dict, servers, current_time)
*/
"""

import math
from scheduler.baseline_schedulers import BaseScheduler
from simulator.models import FunctionInvocation, Tenant, Server


class FairShareScheduler(BaseScheduler):
    """
    Weighted Priority Scheduler with Cost-Efficient Fairness.

    Per-invocation scheduling score:
        Score = (alpha * ThroughputDeficit + beta * SLA_Urgency) / sqrt(base_duration)

    - ThroughputDeficit: arrival-rate-weighted measure of how far below target
      throughput a tenant is. Positive = starved, negative = over-performing.
    - SLA_Urgency: max(latency_urgency, throughput_urgency), capped to [0, 1].
      Reactive safety net that triggers when a tenant approaches SLA violation.
    - sqrt(base_duration): cost-efficiency damping. Gives lightweight functions a
      natural throughput advantage without burying starved tenants' heavy functions.
    """

    def __init__(
        self,
        alpha: float = 0.6,
        beta: float = 0.4,
        sliding_window: float = 1.0,
        sla_latency_threshold: float = 0.500,
        sla_min_throughput_ratio: float = 0.80,
        container_ttl: float = 300.0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.sliding_window = sliding_window
        self.sla_latency_threshold = sla_latency_threshold
        self.sla_min_throughput_ratio = sla_min_throughput_ratio
        self.container_ttl = container_ttl

    def schedule(self, pending_invocations, tenants, servers, current_time):
        if not pending_invocations:
            return []

        # 1. Identify active tenants (those with pending work)
        active_tids = set(inv.tenant_id for inv in pending_invocations)
        active_tenants = {tid: tenants[tid] for tid in active_tids if tid in tenants}
        n_active = len(active_tenants)
        if n_active == 0:
            return []

        # 2. Compute per-tenant deficit and urgency in a single pass
        #    Prune recent_latencies once, then derive both metrics from it.
        window_start = current_time - self.sliding_window
        tenant_priority = {}
        for tid, tenant in active_tenants.items():
            # Prune expired entries once
            while tenant.recent_latencies and tenant.recent_latencies[0][0] < window_start:
                tenant.recent_latencies.popleft()

            n_completed = len(tenant.recent_latencies)
            actual_throughput = n_completed / max(self.sliding_window, 0.001)

            # Throughput deficit
            target_throughput = tenant.arrival_rate * self.sla_min_throughput_ratio
            if target_throughput > 0:
                deficit = (target_throughput - actual_throughput) / target_throughput
                deficit = max(-1.0, min(1.0, deficit))
            else:
                deficit = 0.0

            # SLA urgency — computed from already-pruned deque
            urgency = self._compute_sla_urgency(
                tenant, actual_throughput, n_completed
            )

            tenant_priority[tid] = self.alpha * deficit + self.beta * urgency

        # 3. Pre-compute available server capacity once
        server_avail = {}
        for server in servers:
            server_avail[server.id] = (
                server.cpu_capacity - server.cpu_used,
                server.memory_capacity - server.memory_used,
            )

        # 4. Estimate max schedulable invocations (total free CPU / min CPU demand)
        total_free_cpu = sum(cpu for cpu, _ in server_avail.values())
        min_cpu = min((inv.cpu_demand for inv in pending_invocations), default=100)
        max_schedulable = max(int(total_free_cpu / min_cpu), 1) if min_cpu > 0 else len(pending_invocations)

        # 5. Sort by cost-efficient priority with sqrt damping:
        #    score = tenant_priority / sqrt(base_duration)
        #    sqrt dampens the duration penalty so starved tenants' heavy functions
        #    still beat non-starved tenants' lightweight functions.
        scored = [
            (-(tenant_priority.get(inv.tenant_id, 0.0)
               / math.sqrt(max(inv.base_duration, 0.001))),
             inv.arrival_time, inv)
            for inv in pending_invocations
        ]

        # Partial sort: only fully sort what we can actually schedule
        if max_schedulable < len(scored):
            # Partition around the Nth element, then sort only the top portion
            scored.sort()
            scored = scored[:max_schedulable * 2]  # keep 2x buffer for capacity misses
        else:
            scored.sort()

        # 6. Assign to servers with warm-container + least-loaded preference
        assignments = []
        provisional = {}  # {server_id: (cpu, mem)}

        for _, _, inv in scored:
            server = self._select_server(
                servers, inv.function_type, inv.cpu_demand, inv.memory_demand,
                current_time, provisional, server_avail,
            )
            if server:
                assignments.append((inv, server))
                prev_cpu, prev_mem = provisional.get(server.id, (0, 0))
                provisional[server.id] = (
                    prev_cpu + inv.cpu_demand,
                    prev_mem + inv.memory_demand,
                )

        return assignments

    def _compute_sla_urgency(
        self, tenant: Tenant, actual_throughput: float, n_completed: int
    ) -> float:
        """
        SLA_Urgency = max(latency_urgency, throughput_urgency), capped to [0, 1].
        Deque is already pruned by caller — no re-pruning needed.
        """
        # Latency urgency: ramps from 0 to 1.0 as P95 approaches threshold
        latency_urgency = 0.0
        if n_completed > 0:
            # Manual P95 via sorted index — avoids numpy array allocation
            latencies = sorted(lat for _, lat in tenant.recent_latencies)
            idx = max(0, int(math.ceil(0.95 * len(latencies))) - 1)
            p95 = latencies[idx]
            latency_urgency = min(1.0, p95 / self.sla_latency_threshold)

        # Throughput urgency: how far below minimum guarantee (0-1)
        throughput_urgency = 0.0
        expected = tenant.arrival_rate * self.sla_min_throughput_ratio
        if expected > 0:
            throughput_urgency = min(1.0, max(0.0, 1.0 - actual_throughput / expected))

        return max(latency_urgency, throughput_urgency)

    def _select_server(
        self,
        servers: list[Server],
        function_type: str,
        cpu: int,
        mem: int,
        current_time: float,
        provisional: dict,
        server_avail: dict,
    ) -> Server | None:
        """
        Two-step server selection:
        1. Prefer servers with warm container + capacity → least CPU loaded
        2. Fallback: any server with capacity → least CPU loaded, tie-break by memory
        Uses pre-computed server_avail to skip obviously-full servers fast.
        """
        warm_best = None
        warm_best_key = None
        cold_best = None
        cold_best_key = None

        for server in servers:
            # Quick capacity check using pre-computed availability
            base_cpu, base_mem = server_avail[server.id]
            prov_cpu, prov_mem = provisional.get(server.id, (0, 0))
            avail_cpu = base_cpu - prov_cpu
            avail_mem = base_mem - prov_mem
            if avail_cpu < cpu or avail_mem < mem:
                continue

            eff_cpu_util = (server.cpu_used + prov_cpu) / server.cpu_capacity
            eff_mem_util = (server.memory_used + prov_mem) / server.memory_capacity
            key = (eff_cpu_util, eff_mem_util)

            if server.has_warm_container(function_type, current_time, self.container_ttl):
                if warm_best is None or key < warm_best_key:
                    warm_best = server
                    warm_best_key = key
            else:
                if cold_best is None or key < cold_best_key:
                    cold_best = server
                    cold_best_key = key

        return warm_best if warm_best is not None else cold_best
