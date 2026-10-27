"""Per-packet routing contract for adapted external baselines.

This is deliberately separate from the legacy (node, destination) grouped
router.  It reuses the physical topology, shared FIFO queues, arrival timing,
and 36-packet-per-node service of :mod:`environment`.
"""

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from .config import SimulationConfig
from .environment import LEORoutingEnvironment


Position = Tuple[float, float, float]
TrafficTrace = Tuple[Tuple[Tuple[int, int], ...], ...]


@dataclass(frozen=True)
class PacketDecision:
    packet_id: int
    current: int
    destination: int
    hops: int
    slot: int
    feasible_neighbors: Tuple[int, ...]


@dataclass(frozen=True)
class RoutingSnapshot:
    slot: int
    positions: Tuple[Position, ...]
    queue_lengths: Tuple[int, ...]
    neighbors: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class PacketTransitionEvent:
    decision: PacketDecision
    action: int
    queue_wait_slots: int
    delivered: bool
    dropped: bool
    next_node: int


class PerPacketRouter(Protocol):
    def route_packets(
        self, decisions: Sequence[PacketDecision], snapshot: RoutingSnapshot
    ) -> Dict[int, int]: ...

    def on_slot_end(
        self,
        events: Sequence[PacketTransitionEvent],
        before: RoutingSnapshot,
        after: RoutingSnapshot,
    ) -> None: ...


def direction_slots(node: int, neighbors: Sequence[int], config: SimulationConfig) -> Tuple[Optional[int], ...]:
    """Stable four-way action order: orbit -/+, plane -/+ (with wrap)."""
    plane = node // config.satellites_per_plane
    sat = node % config.satellites_per_plane
    base = plane * config.satellites_per_plane
    predecessor = base + (sat - 1) % config.satellites_per_plane
    successor = base + (sat + 1) % config.satellites_per_plane
    previous_plane = (plane - 1) % config.orbital_planes
    next_plane = (plane + 1) % config.orbital_planes
    inter_previous = sorted(peer for peer in neighbors if peer // config.satellites_per_plane == previous_plane)
    inter_next = sorted(peer for peer in neighbors if peer // config.satellites_per_plane == next_plane)
    result: Tuple[Optional[int], ...] = (
        predecessor if predecessor in neighbors else None,
        successor if successor in neighbors else None,
        inter_previous[0] if inter_previous else None,
        inter_next[0] if inter_next else None,
    )
    present = [peer for peer in result if peer is not None]
    if len(present) != len(set(present)) or set(present) != set(neighbors):
        raise ValueError(f"The four action slots do not cover node {node}: {neighbors}")
    return result


def make_training_trace(
    config: SimulationConfig, load_mbps: float, traffic_seed: int, slots: int
) -> TrafficTrace:
    """Pre-generate exogenous NHPP packets, independent of route outcomes."""
    if slots < 1 or load_mbps < 0:
        raise ValueError("slots must be positive and load_mbps nonnegative")
    rng = random.Random(traffic_seed)
    mean = load_mbps * 1_000_000 * config.slot_duration_s / (config.packet_size_bytes * 8)
    mean /= config.num_satellites
    trace: List[Tuple[Tuple[int, int], ...]] = []
    for slot in range(slots):
        arrivals = []
        for source in range(config.num_satellites):
            phase = 2 * math.pi * source / config.num_satellites
            modulation = 1 + 0.35 * math.sin(2 * math.pi * slot / config.nhpp_period_steps + phase)
            rate = max(0.0, mean * modulation)
            count = 0
            if rate > 0:
                threshold = math.exp(-rate)
                product = 1.0
                while product > threshold:
                    count += 1
                    product *= rng.random()
                count -= 1
            for _ in range(count):
                destination = rng.randrange(config.num_satellites - 1)
                if destination >= source:
                    destination += 1
                arrivals.append((source, destination))
        trace.append(tuple(arrivals))
    return tuple(trace)


class PerPacketRoutingEnvironment(LEORoutingEnvironment):
    """Shared physical environment with stable packet IDs and route events."""

    def __init__(self, config: SimulationConfig, load_mbps: float, trace: TrafficTrace, router: PerPacketRouter):
        super().__init__(config, load_mbps)
        self.trace = trace
        self.router = router
        self.next_packet_id = 0
        self.birth_slot: Dict[int, int] = {}
        self.next_eligible_slot: Dict[int, int] = {}
        self.route_decisions = 0
        self.delivered_delay_slots: List[int] = []
        self.measurement_active = True

    def _generate_background(self) -> None:
        if self.step_index >= len(self.trace):
            raise RuntimeError("Training trace exhausted")
        for source, destination in self.trace[self.step_index]:
            self.next_packet_id += 1
            packet_id = self.next_packet_id
            self.birth_slot[packet_id] = self.step_index
            self.next_eligible_slot[packet_id] = self.step_index
            if self.measurement_active:
                self.background_generated += 1
            self._enqueue(source, (destination, 0, 0, self.measurement_active, packet_id))

    def _record_drop(self, packet, reason: str) -> None:
        super()._record_drop(packet, reason)
        packet_id = packet[4]
        self.birth_slot.pop(packet_id, None)
        self.next_eligible_slot.pop(packet_id, None)

    def _snapshot(self, neighbors: Sequence[Sequence[int]]) -> RoutingSnapshot:
        return RoutingSnapshot(
            slot=self.step_index,
            positions=tuple(self.positions(self.step_index)),
            queue_lengths=tuple(len(queue) for queue in self.node_queues),
            neighbors=tuple(tuple(row) for row in neighbors),
        )

    def _service_nodes(self, neighbors, background_router=None, forced_actions=None):
        if forced_actions:
            raise ValueError("Per-packet environment does not route focal packets")
        before = self._snapshot(neighbors)
        scheduled = []
        decisions = []
        for node, queue in enumerate(self.node_queues):
            for packet in list(queue)[:min(self.config.service_packets_per_slot, len(queue))]:
                scheduled.append((node, packet))
                if packet[2] < self.config.max_hops and neighbors[node]:
                    decisions.append(PacketDecision(
                        packet_id=packet[4], current=node, destination=packet[0],
                        hops=packet[2], slot=self.step_index,
                        feasible_neighbors=tuple(neighbors[node]),
                    ))
        actions = self.router.route_packets(decisions, before) if decisions else {}
        expected = {decision.packet_id for decision in decisions}
        if set(actions) != expected:
            raise ValueError(f"Router returned IDs {set(actions) ^ expected} outside decision contract")
        decisions_by_id = {decision.packet_id: decision for decision in decisions}
        forwarded = []
        event_rows = []
        for node, packet in scheduled:
            actual = self.node_queues[node].popleft()
            if actual != packet:
                raise RuntimeError("Shared FIFO order changed")
            destination, token, hops, measured, packet_id = packet
            if hops >= self.config.max_hops:
                self._record_drop(packet, "ttl")
                continue
            if not neighbors[node]:
                self._record_drop(packet, "no_route")
                continue
            action = actions[packet_id]
            if action not in neighbors[node]:
                raise ValueError(f"Packet {packet_id} chose unavailable ISL {node}->{action}")
            self.route_decisions += 1
            wait = max(0, self.step_index - self.next_eligible_slot[packet_id])
            delivered = action == destination
            if delivered:
                if measured:
                    self.background_delivered += 1
                    self.delivered_delay_slots.append(self.step_index - self.birth_slot[packet_id] + 1)
                self.birth_slot.pop(packet_id)
                self.next_eligible_slot.pop(packet_id)
            else:
                self.next_eligible_slot[packet_id] = self.step_index + 1
                forwarded.append((action, (destination, token, hops + 1, measured, packet_id)))
            event_rows.append((decisions_by_id[packet_id], action, wait, delivered))
        dropped_forwarded = set()
        for target, packet in forwarded:
            if not self._enqueue(target, packet):
                dropped_forwarded.add(packet[4])
        events = [PacketTransitionEvent(
            decision=decision, action=action, queue_wait_slots=wait,
            delivered=delivered,
            dropped=(decision.packet_id in dropped_forwarded or
                     (decision.hops + 1 >= self.config.max_hops and not delivered)),
            next_node=action,
        ) for decision, action, wait, delivered in event_rows]
        after = self._snapshot(neighbors)
        self.router.on_slot_end(events, before, after)
        return set(), set()
