"""
/* USAGE:
  from simulator.models import Tenant, Server, FunctionInvocation, Container

  # Create a server
  server = Server(id="server_01", cpu_capacity=4000, memory_capacity=8192)
  server.has_capacity(500, 512)          # True
  server.allocate(500, 512)
  server.cpu_utilization()               # 0.125

  # Create a tenant
  tenant = Tenant(
      id="tenant_large_01", size="large", arrival_rate=75.0,
      function_profile={"heavy": 0.6, "medium": 0.3, "lightweight": 0.1}
  )

  # Create an invocation
  inv = FunctionInvocation(
      id="inv_tenant_large_01_001", tenant_id="tenant_large_01",
      function_type="heavy", cpu_demand=500, memory_demand=512,
      base_duration=0.15, arrival_time=1.23
  )

  # Create a container
  container = Container(function_type="heavy", server_id="server_01", last_used=10.0)
*/
"""

from dataclasses import dataclass, field
from collections import deque


@dataclass
class FunctionInvocation:
    id: str
    tenant_id: str
    function_type: str        # "lightweight" | "medium" | "heavy"
    cpu_demand: int           # millicores
    memory_demand: int        # MB
    base_duration: float      # seconds (before cold start penalty)
    arrival_time: float       # simulation timestamp
    # Filled during scheduling/execution
    start_time: float = None
    end_time: float = None
    server_id: str = None
    cold_start: bool = False
    wait_time: float = None
    total_latency: float = None


@dataclass
class Container:
    function_type: str
    server_id: str
    last_used: float          # simulation timestamp
    ttl: float = 300.0        # seconds before eviction


@dataclass
class Server:
    id: str
    cpu_capacity: int         # millicores (e.g., 4000)
    memory_capacity: int      # MB (e.g., 8192)
    cpu_used: int = 0
    memory_used: int = 0
    # {function_type: last_used_timestamp}
    warm_containers: dict = field(default_factory=dict)

    def has_capacity(self, cpu: int, mem: int) -> bool:
        return (self.cpu_used + cpu <= self.cpu_capacity and
                self.memory_used + mem <= self.memory_capacity)

    def allocate(self, cpu: int, mem: int):
        self.cpu_used += cpu
        self.memory_used += mem

    def release(self, cpu: int, mem: int):
        self.cpu_used -= cpu
        self.memory_used -= mem

    def cpu_utilization(self) -> float:
        return self.cpu_used / self.cpu_capacity if self.cpu_capacity > 0 else 0.0

    def memory_utilization(self) -> float:
        return self.memory_used / self.memory_capacity if self.memory_capacity > 0 else 0.0

    def has_warm_container(self, function_type: str, current_time: float, ttl: float) -> bool:
        if function_type not in self.warm_containers:
            return False
        return (current_time - self.warm_containers[function_type]) <= ttl

    def refresh_container(self, function_type: str, current_time: float):
        self.warm_containers[function_type] = current_time


@dataclass
class Tenant:
    id: str
    size: str                 # "large" | "medium" | "small"
    arrival_rate: float       # invocations per second
    # e.g., {"heavy": 0.6, "medium": 0.3, "lightweight": 0.1}
    function_profile: dict
    pending_queue: list = field(default_factory=list)
    consumption_window: deque = field(default_factory=deque)
    # Stores recent completed latencies for P95 calculation in FairShareScheduler
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
