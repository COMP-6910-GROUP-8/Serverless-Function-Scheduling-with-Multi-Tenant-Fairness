"""
/* USAGE:
  from scheduler.fairness_scheduler import FairShareScheduler

  scheduler = FairShareScheduler(
      alpha=0.6, beta=0.4, sliding_window=1.0,
      sla_latency_threshold=0.100, sla_min_throughput_ratio=0.80,
      total_cpu_capacity=32000, total_memory_capacity=65536,
  )
  assignments = scheduler.schedule(pending_invocations, tenants_dict, servers, current_time)
*/
"""

import numpy as np
from scheduler.baseline_schedulers import BaseScheduler
from simulator.models import FunctionInvocation, Tenant, Server


class FairShareScheduler(BaseScheduler):
    """
    Weighted Priority Scheduler with Fairness Quotas.
    Priority = alpha * FairShareDeficit + beta * SLA_Urgency
    """

    def __init__(
        self,
        alpha: float = 0.6,
        beta: float = 0.4,
        sliding_window: float = 1.0,
        sla_latency_threshold: float = 0.100,
        sla_min_throughput_ratio: float = 0.80,
        total_cpu_capacity: int = 32000,
        total_memory_capacity: int = 65536,
        container_ttl: float = 300.0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.sliding_window = sliding_window
        self.sla_latency_threshold = sla_latency_threshold
        self.sla_min_throughput_ratio = sla_min_throughput_ratio
        self.total_cpu_capacity = total_cpu_capacity
        self.total_memory_capacity = total_memory_capacity
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

        # 2. Compute per-tenant FairShareDeficit and SLA_Urgency
        # Entitlement in millicore-seconds (rate × window) to match consumption units
        entitlement = (self.total_cpu_capacity / n_active) * self.sliding_window
        window_start = current_time - self.sliding_window

        tenant_priority = {}
        for tid, tenant in active_tenants.items():
            deficit = self._compute_deficit(tenant, entitlement, window_start)
            urgency = self._compute_sla_urgency(tenant, current_time)
            tenant_priority[tid] = self.alpha * deficit + self.beta * urgency

        # 3. Sort invocations by priority (desc), then arrival_time (asc) for ties
        scored = [
            (-tenant_priority.get(inv.tenant_id, 0.0), inv.arrival_time, inv)
            for inv in pending_invocations
        ]
        scored.sort()  # negative priority so highest-first; then earliest arrival

        # 4. Assign to servers with warm-container + least-loaded preference
        assignments = []
        provisional = {}  # {server_id: (cpu, mem)}

        for _, _, inv in scored:
            server = self._select_server(
                servers, inv.function_type, inv.cpu_demand, inv.memory_demand,
                current_time, provisional,
            )
            if server:
                assignments.append((inv, server))
                prev_cpu, prev_mem = provisional.get(server.id, (0, 0))
                provisional[server.id] = (
                    prev_cpu + inv.cpu_demand,
                    prev_mem + inv.memory_demand,
                )

        return assignments

    def _compute_deficit(
        self, tenant: Tenant, entitlement: float, window_start: float
    ) -> float:
        """
        FairShareDeficit = (entitlement - actual) / entitlement
        Positive = starved, Negative = over-consuming.
        """
        # Prune expired entries and sum consumption within window
        actual_cpu = 0.0
        while tenant.consumption_window and tenant.consumption_window[0][0] < window_start:
            tenant.consumption_window.popleft()
        for _, cpu_ms, _ in tenant.consumption_window:
            actual_cpu += cpu_ms

        if entitlement == 0:
            return 0.0
        return (entitlement - actual_cpu) / entitlement

    def _compute_sla_urgency(self, tenant: Tenant, current_time: float) -> float:
        """
        SLA_Urgency = max(latency_urgency, throughput_urgency), capped to [0, 1].
        Uses max so either dimension breaching triggers a response.
        Capped so urgency can't dominate the deficit signal in the priority formula.
        """
        # Latency urgency: 0 if within SLA, ramps to 1.0 at 2x threshold
        latency_urgency = 0.0
        if tenant.recent_latencies:
            p95 = float(np.percentile(list(tenant.recent_latencies), 95))
            latency_urgency = min(1.0, p95 / self.sla_latency_threshold)

        # Throughput urgency: how far below minimum guarantee (already 0-1)
        throughput_urgency = 0.0
        expected = tenant.arrival_rate * self.sla_min_throughput_ratio
        if expected > 0 and tenant.recent_latencies:
            actual_throughput = len(tenant.recent_latencies) / max(
                self.sliding_window, 0.001
            )
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
    ) -> Server | None:
        """
        Two-step server selection:
        1. Prefer servers with warm container + capacity → least CPU loaded
        2. Fallback: any server with capacity → least CPU loaded, tie-break by memory
        """
        warm_candidates = []
        cold_candidates = []

        for server in servers:
            prov_cpu, prov_mem = provisional.get(server.id, (0, 0))
            avail_cpu = server.cpu_capacity - server.cpu_used - prov_cpu
            avail_mem = server.memory_capacity - server.memory_used - prov_mem
            if avail_cpu >= cpu and avail_mem >= mem:
                # Utilization including provisional allocations
                eff_cpu_util = (server.cpu_used + prov_cpu) / server.cpu_capacity
                eff_mem_util = (server.memory_used + prov_mem) / server.memory_capacity
                if server.has_warm_container(
                    function_type, current_time, self.container_ttl
                ):
                    warm_candidates.append((eff_cpu_util, eff_mem_util, server))
                else:
                    cold_candidates.append((eff_cpu_util, eff_mem_util, server))

        # Pick from warm first, then cold — least loaded by CPU, tie-break by memory
        for candidates in [warm_candidates, cold_candidates]:
            if candidates:
                candidates.sort(key=lambda x: (x[0], x[1]))
                return candidates[0][2]

        return None
