import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple

import torch

from .config import SimulationConfig


# destination, focal token (zero for background), traversed hops, measured cohort
Packet = Tuple[int, int, int, bool]
RoutingRequest = Tuple[int, int]
BackgroundRouter = Callable[[Sequence[RoutingRequest], int, List[List[int]]], Dict[RoutingRequest, int]]


@dataclass
class StepResult:
    next_node: int
    reward: float
    delivered: bool
    dropped: bool
    delay_ms: float
    propagation_delay_ms: float
    transmission_delay_ms: float
    queueing_delay_ms: float
    processing_delay_ms: float
    average_queue: float
    simulated_slots: int


class LEORoutingEnvironment:
    """LEO environment with one shared FIFO queue per satellite.

    Each satellite stores at most 100 packets in a shared node queue. The
    calibrated node service rate is shared across all outgoing ISLs. The
    300 Mbps ISL capacity is used only to calculate transmission delay.
    """

    LIGHT_SPEED_KM_S = 299_792.458

    def __init__(self, config: SimulationConfig, offered_load_mbps: float):
        self.config = config
        self.offered_load_mbps = offered_load_mbps
        self._neighbor_cache: Dict[int, List[List[int]]] = {}
        self._token_counter = 0
        self.reset(0)

    def reset(self, traffic_seed: int) -> None:
        self.step_index = 0
        self.rng = random.Random(traffic_seed)
        self.node_queues: List[Deque[Packet]] = [
            deque() for _ in range(self.config.num_satellites)
        ]
        self.measurement_active = False
        self._clear_counters()

    def _clear_counters(self) -> None:
        self.background_generated = 0
        self.background_delivered = 0
        self.background_dropped = 0
        self.background_queue_dropped = 0
        self.background_ttl_dropped = 0
        self.background_no_route_dropped = 0

    def reset_measurement_counters(self) -> None:
        self._clear_counters()
        self.measurement_active = True

    def sample_source_destination(self) -> Tuple[int, int]:
        source = self.rng.randrange(self.config.num_satellites)
        destination = self.rng.randrange(self.config.num_satellites - 1)
        if destination >= source:
            destination += 1
        return source, destination

    def positions(self, step_index: int) -> List[Tuple[float, float, float]]:
        cfg = self.config
        radius = cfg.earth_radius_km + cfg.orbital_altitude_km
        inclination = math.radians(cfg.inclination_deg)
        elapsed_s = step_index * cfg.slot_duration_s
        orbital_phase = (elapsed_s % cfg.orbital_period_s) / cfg.orbital_period_s
        orbital_advance = 2.0 * math.pi * orbital_phase
        positions: List[Tuple[float, float, float]] = []
        for plane in range(cfg.orbital_planes):
            raan = 2.0 * math.pi * plane / cfg.orbital_planes
            phase = 2.0 * math.pi * cfg.phasing_factor * plane / cfg.num_satellites
            for sat in range(cfg.satellites_per_plane):
                anomaly = 2.0 * math.pi * sat / cfg.satellites_per_plane + phase + orbital_advance
                x_orb = radius * math.cos(anomaly)
                y_orb = radius * math.sin(anomaly)
                x = math.cos(raan) * x_orb - math.sin(raan) * math.cos(inclination) * y_orb
                y = math.sin(raan) * x_orb + math.cos(raan) * math.cos(inclination) * y_orb
                z = math.sin(inclination) * y_orb
                positions.append((x, y, z))
        return positions

    @staticmethod
    def distance(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def has_line_of_sight(
        self, a: Sequence[float], b: Sequence[float]
    ) -> bool:
        """Return whether the Earth does not intersect the satellite segment."""
        segment = [b_i - a_i for a_i, b_i in zip(a, b)]
        segment_norm_sq = sum(component * component for component in segment)
        if segment_norm_sq == 0.0:
            return False
        projection = -sum(a_i * component for a_i, component in zip(a, segment))
        fraction = min(1.0, max(0.0, projection / segment_norm_sq))
        closest = [a_i + fraction * component for a_i, component in zip(a, segment)]
        closest_radius = math.sqrt(sum(component * component for component in closest))
        return closest_radius + 1e-9 >= self.config.earth_radius_km

    def _match_adjacent_planes(
        self,
        left_nodes: Sequence[int],
        right_nodes: Sequence[int],
        positions: Sequence[Sequence[float]],
        eligible: Sequence[bool],
    ) -> List[Tuple[int, int]]:
        """Build a deterministic maximum-cardinality one-to-one ISL matching.

        Candidate peers are tried from shortest to longest distance. The
        augmenting-path matching prevents the many-to-one links produced by
        independently selecting each satellite's nearest peer.
        """
        candidates: Dict[int, List[int]] = {}
        for node in left_nodes:
            if not eligible[node]:
                candidates[node] = []
                continue
            peers = [
                peer
                for peer in right_nodes
                if eligible[peer]
                and self.has_line_of_sight(positions[node], positions[peer])
            ]
            candidates[node] = sorted(
                peers,
                key=lambda peer: (self.distance(positions[node], positions[peer]), peer),
            )

        right_to_left: Dict[int, int] = {}

        def assign(node: int, visited: Set[int]) -> bool:
            for peer in candidates[node]:
                if peer in visited:
                    continue
                visited.add(peer)
                incumbent = right_to_left.get(peer)
                if incumbent is None or assign(incumbent, visited):
                    right_to_left[peer] = node
                    return True
            return False

        ordered_left = sorted(
            left_nodes,
            key=lambda node: (
                self.distance(positions[node], positions[candidates[node][0]])
                if candidates[node]
                else math.inf,
                node,
            ),
        )
        for node in ordered_left:
            assign(node, set())
        return sorted((node, peer) for peer, node in right_to_left.items())

    def neighbors_at(self, step_index: int) -> List[List[int]]:
        interval = self.config.topology_update_interval_steps
        cache_key = step_index - step_index % interval
        if cache_key in self._neighbor_cache:
            return self._neighbor_cache[cache_key]
        cfg = self.config
        positions = self.positions(cache_key)
        neighbors = [set() for _ in range(cfg.num_satellites)]

        for plane in range(cfg.orbital_planes):
            base = plane * cfg.satellites_per_plane
            for sat in range(cfg.satellites_per_plane):
                node = base + sat
                for offset in (-1, 1):
                    peer = base + (sat + offset) % cfg.satellites_per_plane
                    neighbors[node].add(peer)
                    neighbors[peer].add(node)

        latitude_limit = cfg.inter_plane_latitude_limit_deg
        eligible = []
        for x, y, z in positions:
            latitude = math.degrees(math.asin(z / math.sqrt(x * x + y * y + z * z)))
            eligible.append(abs(latitude) <= latitude_limit)

        plane_pairs = {
            tuple(sorted((plane, (plane + 1) % cfg.orbital_planes)))
            for plane in range(cfg.orbital_planes)
        }
        for left_plane, right_plane in sorted(plane_pairs):
            left_nodes = list(
                range(
                    left_plane * cfg.satellites_per_plane,
                    (left_plane + 1) * cfg.satellites_per_plane,
                )
            )
            right_nodes = list(
                range(
                    right_plane * cfg.satellites_per_plane,
                    (right_plane + 1) * cfg.satellites_per_plane,
                )
            )
            for node, peer in self._match_adjacent_planes(
                left_nodes, right_nodes, positions, eligible
            ):
                neighbors[node].add(peer)
                neighbors[peer].add(node)

        result = [sorted(items) for items in neighbors]
        self._neighbor_cache[cache_key] = result
        return result

    def adjacency_tensor(self, step_index: int) -> torch.Tensor:
        n = self.config.num_satellites
        matrix = [[False] * n for _ in range(n)]
        for node, peers in enumerate(self.neighbors_at(step_index)):
            matrix[node][node] = True
            for peer in peers:
                matrix[node][peer] = True
        return torch.tensor(matrix, dtype=torch.bool)

    def state_tensor(self, current: int, destination: int, step_index: int) -> torch.Tensor:
        return self.state_batch([(current, destination)], step_index)[0]

    def state_batch(
        self, requests: Sequence[RoutingRequest], step_index: int
    ) -> torch.Tensor:
        positions = self.positions(step_index)
        scale = 2.0 * (
            self.config.earth_radius_km + self.config.orbital_altitude_km
        )
        queue_ratios = [
            len(queue) / self.config.max_queue_packets for queue in self.node_queues
        ]
        batch_features = []
        for current, destination in requests:
            destination_position = positions[destination]
            features = []
            for node, position in enumerate(positions):
                relative = [
                    (position[i] - destination_position[i]) / scale for i in range(3)
                ]
                distance = self.distance(position, destination_position) / scale
                features.append(
                    relative
                    + [queue_ratios[node], distance, float(node == current)]
                )
            batch_features.append(features)
        return torch.tensor(batch_features, dtype=torch.float32)

    def _poisson(self, rate: float) -> int:
        if rate <= 0.0:
            return 0
        threshold = math.exp(-rate)
        product = 1.0
        count = 0
        while product > threshold:
            count += 1
            product *= self.rng.random()
        return count - 1

    def _record_drop(self, packet: Packet, reason: str) -> None:
        if packet[1] != 0 or not packet[3]:
            return
        self.background_dropped += 1
        if reason == "queue":
            self.background_queue_dropped += 1
        elif reason == "ttl":
            self.background_ttl_dropped += 1
        elif reason == "no_route":
            self.background_no_route_dropped += 1

    def _enqueue(self, node: int, packet: Packet) -> bool:
        queue = self.node_queues[node]
        if len(queue) >= self.config.max_queue_packets:
            self._record_drop(packet, "queue")
            return False
        queue.append(packet)
        return True

    def _generate_background(self) -> None:
        cfg = self.config
        packets_per_slot = (
            self.offered_load_mbps * 1_000_000.0 * cfg.slot_duration_s
            / (cfg.packet_size_bytes * 8.0)
        )
        base_rate = packets_per_slot / cfg.num_satellites
        for source in range(cfg.num_satellites):
            phase = 2.0 * math.pi * source / cfg.num_satellites
            modulation = 1.0 + 0.35 * math.sin(
                2.0 * math.pi * self.step_index / cfg.nhpp_period_steps + phase
            )
            arrivals = self._poisson(max(0.0, base_rate * modulation))
            for _ in range(arrivals):
                destination = self.rng.randrange(cfg.num_satellites - 1)
                if destination >= source:
                    destination += 1
                packet = (destination, 0, 0, self.measurement_active)
                if self.measurement_active:
                    self.background_generated += 1
                self._enqueue(source, packet)

    def _shortest_next_hop(
        self, source: int, destination: int, neighbors: List[List[int]]
    ) -> Optional[int]:
        if source == destination:
            return source
        frontier: Deque[int] = deque([source])
        previous = {source: -1}
        while frontier:
            node = frontier.popleft()
            for peer in neighbors[node]:
                if peer in previous:
                    continue
                previous[peer] = node
                if peer == destination:
                    cursor = destination
                    while previous[cursor] != source:
                        cursor = previous[cursor]
                    return cursor
                frontier.append(peer)
        return None

    def _service_nodes(
        self,
        neighbors: List[List[int]],
        background_router: Optional[BackgroundRouter],
        forced_actions: Optional[Dict[int, int]] = None,
    ) -> Tuple[Set[int], Set[int]]:
        forced_actions = forced_actions or {}
        scheduled: List[Tuple[int, Packet]] = []
        requests: List[RoutingRequest] = []
        seen_requests = set()
        for node, queue in enumerate(self.node_queues):
            service_count = min(self.config.service_packets_per_slot, len(queue))
            packets = list(queue)[:service_count]
            for packet in packets:
                scheduled.append((node, packet))
                request = (node, packet[0])
                if packet[1] == 0 and packet[2] < self.config.max_hops and request not in seen_requests:
                    requests.append(request)
                    seen_requests.add(request)

        policy_actions = (
            background_router(requests, self.step_index, neighbors)
            if background_router is not None and requests
            else {}
        )
        forwarded: List[Tuple[int, Packet]] = []
        served_focal: Set[int] = set()
        dropped_focal: Set[int] = set()

        for node, packet in scheduled:
            actual = self.node_queues[node].popleft()
            if actual != packet:
                raise RuntimeError("Shared FIFO queue order changed during service")
            destination, token, hops, measured = packet
            if token == 0 and hops >= self.config.max_hops:
                self._record_drop(packet, "ttl")
                continue
            action = forced_actions.get(token) if token != 0 else policy_actions.get((node, destination))
            if action not in neighbors[node]:
                if token != 0:
                    dropped_focal.add(token)
                    continue
                action = self._shortest_next_hop(node, destination, neighbors)
            if action is None:
                self._record_drop(packet, "no_route")
                continue
            if token != 0:
                served_focal.add(token)
            elif action == destination:
                if measured:
                    self.background_delivered += 1
            else:
                forwarded.append((action, (destination, 0, hops + 1, measured)))

        for target, packet in forwarded:
            self._enqueue(target, packet)
        return served_focal, dropped_focal

    def average_node_queue(self) -> float:
        return sum(len(queue) for queue in self.node_queues) / len(self.node_queues)

    def advance_background(self, background_router: Optional[BackgroundRouter] = None) -> float:
        neighbors = self.neighbors_at(self.step_index)
        self._generate_background()
        self._service_nodes(neighbors, background_router)
        average_queue = self.average_node_queue()
        self.step_index += 1
        return average_queue

    def _remove_focal(self, focal_token: int) -> None:
        for node, queue in enumerate(self.node_queues):
            kept = deque(packet for packet in queue if packet[1] != focal_token)
            self.node_queues[node] = kept

    def step(
        self,
        current: int,
        action: int,
        destination: int,
        hop_index: int,
        background_router: Optional[BackgroundRouter] = None,
        max_simulated_slots: Optional[int] = None,
    ) -> StepResult:
        cfg = self.config
        neighbors = self.neighbors_at(self.step_index)
        if action not in neighbors[current]:
            raise ValueError(f"Action {action} is not a feasible neighbor of node {current}")

        positions = self.positions(self.step_index)
        link_distance = self.distance(positions[current], positions[action])
        previous_distance = self.distance(positions[current], positions[destination])
        next_distance = self.distance(positions[action], positions[destination])
        propagation = link_distance / self.LIGHT_SPEED_KM_S * 1000.0
        transmission = (
            cfg.packet_size_bytes * 8.0 / (cfg.isl_capacity_mbps * 1_000_000.0) * 1000.0
        )
        processing = cfg.processing_delay_ms

        self._generate_background()
        queue_ahead = len(self.node_queues[current])
        queueing = (
            queue_ahead / cfg.service_packets_per_slot * cfg.slot_duration_s * 1000.0
        )
        self._token_counter += 1
        focal_token = self._token_counter
        focal_dropped = not self._enqueue(
            current, (destination, focal_token, hop_index, False)
        )

        served_tokens: Set[int] = set()
        dropped_tokens: Set[int] = set()
        if not focal_dropped:
            served_tokens, dropped_tokens = self._service_nodes(
                neighbors,
                background_router,
                forced_actions={focal_token: action},
            )
            focal_dropped = focal_token in dropped_tokens

        simulated_slots = 1
        limit = math.ceil(cfg.max_queue_packets / cfg.service_packets_per_slot) + 2
        if max_simulated_slots is not None:
            limit = min(limit, max(1, max_simulated_slots))
        while not focal_dropped and focal_token not in served_tokens:
            if simulated_slots >= limit:
                focal_dropped = True
                self._remove_focal(focal_token)
                break
            self.step_index += 1
            neighbors = self.neighbors_at(self.step_index)
            self._generate_background()
            served_tokens, dropped_tokens = self._service_nodes(
                neighbors,
                background_router,
                forced_actions={focal_token: action},
            )
            focal_dropped = focal_token in dropped_tokens
            simulated_slots += 1

        average_queue = self.average_node_queue()
        self.step_index += 1
        delivered = not focal_dropped and action == destination
        ttl_expired = hop_index + 1 >= cfg.max_hops and not delivered
        dropped = focal_dropped or ttl_expired

        distance_reward = 1.0 if next_distance < previous_distance else -1.0
        congestion_penalty = 2.0 * min(1.0, queue_ahead / cfg.max_queue_packets)
        reward = distance_reward - congestion_penalty - 0.1
        if delivered:
            reward = 10.0
        elif dropped:
            reward = -10.0

        return StepResult(
            next_node=action,
            reward=reward,
            delivered=delivered,
            dropped=dropped,
            delay_ms=propagation + transmission + queueing + processing,
            propagation_delay_ms=propagation,
            transmission_delay_ms=transmission,
            queueing_delay_ms=queueing,
            processing_delay_ms=processing,
            average_queue=average_queue,
            simulated_slots=simulated_slots,
        )

    def background_metrics(self, measured_slots: int) -> Dict[str, float]:
        duration_s = max(1, measured_slots) * self.config.slot_duration_s
        throughput = (
            self.background_delivered
            * self.config.packet_size_bytes
            * 8.0
            / duration_s
            / 1_000_000.0
        )
        loss_rate = (
            self.background_dropped / self.background_generated
            if self.background_generated > 0
            else 0.0
        )
        return {
            "throughput_mbps": throughput,
            "packet_loss": loss_rate,
            "background_generated": float(self.background_generated),
            "background_delivered": float(self.background_delivered),
            "background_dropped": float(self.background_dropped),
            "background_queue_dropped": float(self.background_queue_dropped),
            "background_ttl_dropped": float(self.background_ttl_dropped),
            "background_no_route_dropped": float(self.background_no_route_dropped),
        }
