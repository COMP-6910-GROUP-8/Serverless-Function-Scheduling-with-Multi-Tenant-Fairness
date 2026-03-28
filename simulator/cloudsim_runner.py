"""
/* USAGE:
  from simulator.cloudsim_runner import SimulationEngine
  from scheduler import FIFOScheduler

  engine = SimulationEngine(config, scheduler=FIFOScheduler(), verbose=False)
  completed, overheads = engine.run(tenants, invocations)
  # completed: list[FunctionInvocation] with all fields filled
  # overheads: list[float] — wall-clock seconds per schedule() call
*/
"""

import time
import simpy
from scheduler.baseline_schedulers import BaseScheduler
from simulator.models import FunctionInvocation, Tenant, Server

# Scheduling tick interval (seconds of simulation time).
# The scheduler runs at most once per tick, batching arrivals and completions.
SCHEDULE_TICK = 0.010  # 10ms


class SimulationEngine:
    def __init__(self, config: dict, scheduler: BaseScheduler, verbose: bool = False):
        self.config = config
        self.scheduler = scheduler
        self.verbose = verbose

        self.cold_start_penalty = config["cold_start"]["penalty"]
        self.container_ttl = config["cold_start"]["container_ttl"]

        server_cfg = config["servers"]
        self.servers = [
            Server(
                id=f"server_{i+1:02d}",
                cpu_capacity=server_cfg["cpu_capacity"],
                memory_capacity=server_cfg["memory_capacity"],
            )
            for i in range(server_cfg["count"])
        ]

    def run(
        self, tenants: list[Tenant], invocations: list[FunctionInvocation]
    ) -> tuple[list[FunctionInvocation], list[float]]:
        self.completed: list[FunctionInvocation] = []
        self.scheduling_overheads: list[float] = []
        self.tenant_map = {t.id: t for t in tenants}
        self._dirty = False  # True when pending queues have changed since last schedule
        duration = self.config["experiment"]["duration"]

        # Reset server state
        for s in self.servers:
            s.cpu_used = 0
            s.memory_used = 0
            s.warm_containers.clear()

        # Reset tenant runtime state
        for t in tenants:
            t.pending_queue.clear()
            t.consumption_window.clear()
            t.recent_latencies.clear()

        env = simpy.Environment()
        self.env = env

        # Start background processes
        env.process(self._container_eviction(env))
        env.process(self._scheduler_tick(env, duration))
        env.process(self._invocation_arrival(env, invocations, duration))

        env.run(until=duration)

        return self.completed, self.scheduling_overheads

    def _invocation_arrival(self, env, invocations, duration):
        """Feed invocations into the simulation at their arrival times."""
        for inv in invocations:
            if inv.arrival_time >= duration:
                break
            delay = inv.arrival_time - env.now
            if delay > 0:
                yield env.timeout(delay)

            tenant = self.tenant_map.get(inv.tenant_id)
            if tenant:
                tenant.pending_queue.append(inv)
                self._dirty = True

    def _scheduler_tick(self, env, duration):
        """Run the scheduler at fixed intervals, batching work between ticks."""
        while env.now < duration:
            yield env.timeout(SCHEDULE_TICK)
            if self._dirty:
                self._run_schedule(env)

    def _run_schedule(self, env):
        """Collect all pending invocations, call scheduler, dispatch assignments."""
        all_pending = []
        for tenant in self.tenant_map.values():
            if tenant.pending_queue:
                all_pending.extend(tenant.pending_queue)

        if not all_pending:
            self._dirty = False
            return

        # Measure scheduling overhead
        t0 = time.perf_counter()
        assignments = self.scheduler.schedule(
            all_pending, self.tenant_map, self.servers, env.now
        )
        t1 = time.perf_counter()
        self.scheduling_overheads.append(t1 - t0)

        if not assignments:
            return

        # Remove assigned invocations from their tenant queues and start execution
        assigned_ids = {inv.id for inv, _ in assignments}
        for inv, server in assignments:
            env.process(self._execute_invocation(env, inv, server))

        for tenant in self.tenant_map.values():
            if tenant.pending_queue:
                tenant.pending_queue = [
                    i for i in tenant.pending_queue if i.id not in assigned_ids
                ]

        self._dirty = any(t.pending_queue for t in self.tenant_map.values())

    def _execute_invocation(self, env, inv: FunctionInvocation, server: Server):
        """Execute a single function invocation on the assigned server."""
        inv.start_time = env.now
        inv.server_id = server.id

        # Check warm container
        inv.cold_start = not server.has_warm_container(
            inv.function_type, env.now, self.container_ttl
        )
        execution_time = inv.base_duration + (
            self.cold_start_penalty if inv.cold_start else 0.0
        )

        # Reserve resources
        server.allocate(inv.cpu_demand, inv.memory_demand)

        if self.verbose:
            cs_label = "COLD" if inv.cold_start else "WARM"
            print(
                f"  [{env.now:.3f}s] {inv.tenant_id} -> {server.id} "
                f"({inv.function_type}, {cs_label}, {execution_time*1000:.1f}ms)"
            )

        # Simulate execution
        yield env.timeout(execution_time)

        # Release resources
        server.release(inv.cpu_demand, inv.memory_demand)
        server.refresh_container(inv.function_type, env.now)

        # Record results
        inv.end_time = env.now
        inv.wait_time = inv.start_time - inv.arrival_time
        inv.total_latency = inv.end_time - inv.arrival_time

        # Update tenant tracking for FairShareScheduler
        tenant = self.tenant_map.get(inv.tenant_id)
        if tenant:
            cpu_ms = inv.cpu_demand * execution_time
            mem_mb_ms = inv.memory_demand * execution_time
            tenant.consumption_window.append((env.now, cpu_ms, mem_mb_ms))
            tenant.recent_latencies.append((env.now, inv.total_latency))

        self.completed.append(inv)
        self._dirty = True  # freed resources, pending work may now fit

    def _container_eviction(self, env):
        """Background process: evict expired warm containers every 1 second."""
        while True:
            yield env.timeout(1.0)
            for server in self.servers:
                expired = [
                    ftype for ftype, last_used in server.warm_containers.items()
                    if env.now - last_used > self.container_ttl
                ]
                for ftype in expired:
                    del server.warm_containers[ftype]
