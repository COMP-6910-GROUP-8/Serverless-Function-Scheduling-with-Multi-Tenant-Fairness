"""
/* USAGE:
  from scheduler.baseline_schedulers import FIFOScheduler, RoundRobinScheduler, SJFScheduler

  scheduler = FIFOScheduler()
  assignments = scheduler.schedule(pending_invocations, tenants_dict, servers, current_time)
  # assignments is list of (FunctionInvocation, Server) tuples
*/
"""

from abc import ABC, abstractmethod
from simulator.models import FunctionInvocation, Tenant, Server


class BaseScheduler(ABC):
    @abstractmethod
    def schedule(
        self,
        pending_invocations: list[FunctionInvocation],
        tenants: dict[str, Tenant],
        servers: list[Server],
        current_time: float,
    ) -> list[tuple[FunctionInvocation, Server]]:
        """Return list of (invocation, server) assignments."""
        pass


def _find_available_server(
    servers: list[Server],
    cpu: int,
    mem: int,
    provisional: dict[str, tuple[int, int]],
) -> Server | None:
    """
    Find first server with enough capacity after accounting for provisional
    allocations made earlier in the same schedule() call.
    """
    for server in servers:
        prov_cpu, prov_mem = provisional.get(server.id, (0, 0))
        if (server.cpu_used + prov_cpu + cpu <= server.cpu_capacity and
                server.memory_used + prov_mem + mem <= server.memory_capacity):
            return server
    return None


def _provision(provisional: dict, server: Server, cpu: int, mem: int):
    """Track a provisional allocation so later invocations see updated capacity."""
    prev_cpu, prev_mem = provisional.get(server.id, (0, 0))
    provisional[server.id] = (prev_cpu + cpu, prev_mem + mem)


class FIFOScheduler(BaseScheduler):
    """First-In-First-Out: processes invocations in arrival order."""

    def schedule(self, pending_invocations, tenants, servers, current_time):
        sorted_inv = sorted(pending_invocations, key=lambda i: i.arrival_time)
        assignments = []
        provisional = {}  # {server_id: (cpu_reserved, mem_reserved)}

        for inv in sorted_inv:
            server = _find_available_server(
                servers, inv.cpu_demand, inv.memory_demand, provisional
            )
            if server:
                assignments.append((inv, server))
                _provision(provisional, server, inv.cpu_demand, inv.memory_demand)

        return assignments


class RoundRobinScheduler(BaseScheduler):
    """Cycles through tenants, picking one invocation per tenant per round."""

    def __init__(self):
        self._tenant_index = 0

    def schedule(self, pending_invocations, tenants, servers, current_time):
        # Group pending by tenant, keeping arrival order within each tenant
        per_tenant = {}
        for inv in pending_invocations:
            per_tenant.setdefault(inv.tenant_id, []).append(inv)
        for tid in per_tenant:
            per_tenant[tid].sort(key=lambda i: i.arrival_time)

        active_tids = list(per_tenant.keys())
        if not active_tids:
            return []

        assignments = []
        provisional = {}
        # Wrap index to valid range
        self._tenant_index = self._tenant_index % len(active_tids)
        stalled = 0  # tracks consecutive tenants we couldn't assign

        while stalled < len(active_tids) and active_tids:
            tid = active_tids[self._tenant_index % len(active_tids)]
            queue = per_tenant.get(tid, [])

            assigned = False
            if queue:
                inv = queue[0]
                server = _find_available_server(
                    servers, inv.cpu_demand, inv.memory_demand, provisional
                )
                if server:
                    assignments.append((inv, server))
                    _provision(provisional, server, inv.cpu_demand, inv.memory_demand)
                    queue.pop(0)
                    assigned = True
                    # Remove tenant from active list if no more pending
                    if not queue:
                        active_tids.remove(tid)
                        if not active_tids:
                            break
                        # Don't advance index since list shifted
                        stalled = 0
                        continue

            if assigned:
                stalled = 0
            else:
                stalled += 1

            self._tenant_index = (self._tenant_index + 1) % max(len(active_tids), 1)

        return assignments


class SJFScheduler(BaseScheduler):
    """Shortest-Job-First: prioritizes invocations with shortest duration."""

    def schedule(self, pending_invocations, tenants, servers, current_time):
        sorted_inv = sorted(pending_invocations, key=lambda i: i.base_duration)
        assignments = []
        provisional = {}

        for inv in sorted_inv:
            server = _find_available_server(
                servers, inv.cpu_demand, inv.memory_demand, provisional
            )
            if server:
                assignments.append((inv, server))
                _provision(provisional, server, inv.cpu_demand, inv.memory_demand)

        return assignments
