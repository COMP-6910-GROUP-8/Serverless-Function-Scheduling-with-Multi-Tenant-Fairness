"""
/* USAGE:
  from scheduler.fairness_scheduler import FairShareScheduler

  scheduler = FairShareScheduler(
      sliding_window=5.0,
      container_ttl=300.0,
  )
  assignments = scheduler.schedule(pending_invocations, tenants_dict, servers, current_time)
*/
"""

from scheduler.baseline_schedulers import BaseScheduler
from simulator.models import FunctionInvocation, Tenant, Server


class FairShareScheduler(BaseScheduler):
    """
    Fair-Share Scheduler — two-phase scheduling.

    Phase 1 (Fairness): Each active tenant gets at least one dispatch per
        scheduling round, selected by deficit (most under-served first).
        Within a tenant, the shortest job is picked (SJF).
    Phase 2 (Efficiency): Remaining capacity filled via global SJF.
    """

    def __init__(
        self,
        sliding_window: float = 5.0,
        container_ttl: float = 300.0,
    ):
        self.sliding_window = sliding_window
        self.container_ttl = container_ttl
        self._dispatch_counts: dict[str, int] = {}
        self._window_start: float = 0.0
        self._total_dispatched: int = 0

    def schedule(self, pending_invocations, tenants, servers, current_time):
        if not pending_invocations:
            return []

        # Reset sliding window if expired
        if current_time - self._window_start >= self.sliding_window:
            self._dispatch_counts.clear()
            self._total_dispatched = 0
            self._window_start = current_time

        per_tenant: dict[str, list[FunctionInvocation]] = {}
        for inv in pending_invocations:
            per_tenant.setdefault(inv.tenant_id, []).append(inv)
        for tid in per_tenant:
            per_tenant[tid].sort(key=lambda i: (i.base_duration, i.arrival_time))

        server_avail: dict[str, tuple[int, int]] = {}
        for server in servers:
            server_avail[server.id] = (
                server.cpu_capacity - server.cpu_used,
                server.memory_capacity - server.memory_used,
            )

        assignments = []
        provisional: dict[str, tuple[int, int]] = {}
        assigned_ids: set[str] = set()

        # Phase 1: Fairness guarantee
        active_tids = list(per_tenant.keys())
        n_active = len(active_tids)
        if n_active > 0:
            fair_share = self._total_dispatched / n_active
            tenant_deficits = []
            for tid in active_tids:
                dispatched = self._dispatch_counts.get(tid, 0)
                deficit = fair_share - dispatched
                oldest = per_tenant[tid][0].arrival_time
                tenant_deficits.append((deficit, -oldest, tid))
            tenant_deficits.sort(key=lambda x: (-x[0], x[1]))

            for _, _, tid in tenant_deficits:
                inv = per_tenant[tid][0]
                server = self._select_server(
                    servers, inv.function_type, inv.cpu_demand, inv.memory_demand,
                    current_time, provisional, server_avail,
                )
                if server:
                    assignments.append((inv, server))
                    assigned_ids.add(inv.id)
                    self._track_provisional(provisional, server, inv)
                    self._track_dispatch(tid)

        # Phase 2: Fill remaining capacity with global SJF
        remaining = [
            inv for inv in pending_invocations
            if inv.id not in assigned_ids
        ]
        remaining.sort(key=lambda i: (i.base_duration, i.arrival_time))

        for inv in remaining:
            server = self._select_server(
                servers, inv.function_type, inv.cpu_demand, inv.memory_demand,
                current_time, provisional, server_avail,
            )
            if server:
                assignments.append((inv, server))
                assigned_ids.add(inv.id)
                self._track_provisional(provisional, server, inv)
                self._track_dispatch(inv.tenant_id)

        return assignments

    def _track_provisional(self, provisional, server, inv):
        prev_cpu, prev_mem = provisional.get(server.id, (0, 0))
        provisional[server.id] = (prev_cpu + inv.cpu_demand, prev_mem + inv.memory_demand)

    def _track_dispatch(self, tenant_id):
        self._dispatch_counts[tenant_id] = self._dispatch_counts.get(tenant_id, 0) + 1
        self._total_dispatched += 1

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
        """Select server: prefer warm container, then least-loaded. Tie-break by memory."""
        warm_best = None
        warm_best_key = None
        cold_best = None
        cold_best_key = None

        for server in servers:
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
