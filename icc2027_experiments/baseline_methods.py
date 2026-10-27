"""Adapted GraphPR, FDR-MARL, and DQN-IR policies for the common simulator.

These are research adaptations described in ICC2027_BASELINE_ADAPTATION_SPEC.md,
not implementations or checkpoints distributed by the original authors.
"""

import math
import random
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .baseline_packet import (
    PacketDecision, PacketTransitionEvent, RoutingSnapshot, direction_slots,
)
from .config import BASELINE_LEARNING, SimulationConfig


BASELINE_NAMES = ("graphpr_adapted", "fdr_marl_adapted", "dqn_ir_fixed_link")
GRAPHPR_INPUT = 7 + 4 * 256 + 4 + 6
FDR_INPUT = 4 + 4 * 6
DQN_IR_INPUT = 4 * 5


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _shortest_hops(
    start: int, destination: int, graph: Sequence[Sequence[int]], blocked: Optional[int] = None
) -> Optional[int]:
    if start == blocked or destination == blocked:
        return None
    if start == destination:
        return 0
    queue = deque([(start, 0)])
    seen = {start, blocked}
    while queue:
        node, depth = queue.popleft()
        for peer in graph[node]:
            if peer in seen:
                continue
            if peer == destination:
                return depth + 1
            seen.add(peer)
            queue.append((peer, depth + 1))
    return None


def residual_shortest_path_hops(
    current: int, neighbor: int, destination: int,
    graph: Sequence[Sequence[int]], max_hops: int = 30,
) -> int:
    current_hops = _shortest_hops(current, destination, graph)
    remaining = _shortest_hops(neighbor, destination, graph, blocked=current)
    if current_hops is None or remaining is None:
        return -max_hops
    return current_hops - remaining


def grid_manhattan_distance(node: int, destination: int, config: SimulationConfig) -> int:
    plane_a, sat_a = divmod(node, config.satellites_per_plane)
    plane_b, sat_b = divmod(destination, config.satellites_per_plane)
    plane_gap = abs(plane_a - plane_b)
    sat_gap = abs(sat_a - sat_b)
    return min(plane_gap, config.orbital_planes - plane_gap) + min(
        sat_gap, config.satellites_per_plane - sat_gap
    )


class GraphPRQ(nn.Module):
    """One local attention layer plus four affine FCNN layers."""

    def __init__(self):
        super().__init__()
        self.local = nn.Linear(7, 256)
        self.neighbor = nn.Linear(256, 256)
        self.query = nn.Linear(256, 256)
        self.key = nn.Linear(256, 256)
        self.score = nn.Linear(256, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(256 + 6, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 4),
        )

    def encode(self, local: torch.Tensor, neighbor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        own = F.relu(self.local(local))
        others = F.relu(self.neighbor(neighbor))
        values = torch.cat((own.unsqueeze(1), others), dim=1)
        query = self.query(own).unsqueeze(1)
        score = self.score(F.leaky_relu(query + self.key(values), negative_slope=0.2)).squeeze(-1)
        full_mask = torch.cat((torch.ones_like(mask[:, :1]), mask), dim=1).bool()
        score = score.masked_fill(~full_mask, torch.finfo(score.dtype).min)
        weights = F.softmax(score, dim=1)
        return F.elu((weights.unsqueeze(-1) * values).sum(dim=1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        local = state[:, :7]
        neighbor = state[:, 7:7 + 4 * 256].reshape(-1, 4, 256)
        mask = state[:, 7 + 4 * 256:7 + 4 * 256 + 4]
        packet = state[:, -6:]
        return self.head(torch.cat((self.encode(local, neighbor, mask), packet), dim=1))


class FDRQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FDR_INPUT, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(128, 128)
        self.second = nn.Linear(128, 128)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.relu(state + self.second(F.relu(self.first(state))))


class DQNIRResNet(nn.Module):
    """Eight affine layers: input, three two-layer blocks, output."""

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(DQN_IR_INPUT, 128)
        self.blocks = nn.Sequential(ResidualBlock(), ResidualBlock(), ResidualBlock())
        self.output = nn.Linear(128, 4)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(F.relu(self.input(state))))


class AdaptedBaselinePolicy:
    """Policy, per-agent replay, Q-learning, and per-packet transition collector."""

    def __init__(
        self, name: str, config: SimulationConfig, seed: int, *, training: bool = True,
        pilot_batch_size: Optional[int] = None,
        device: str = "cpu",
        update_every_slots: int = 1,
        epsilon_decay_decisions: Optional[int] = None,
    ):
        if name not in BASELINE_NAMES:
            raise ValueError(f"Unknown adapted baseline {name}")
        self.name = name
        self.config = config
        self.training = training
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.agent_count = config.num_satellites if name != "dqn_ir_fixed_link" else 1
        factory = {
            "graphpr_adapted": GraphPRQ,
            "fdr_marl_adapted": FDRQ,
            "dqn_ir_fixed_link": DQNIRResNet,
        }[name]
        self.models = [factory().to(self.device) for _ in range(self.agent_count)]
        self.targets = [factory().to(self.device) for _ in range(self.agent_count)]
        for model, target in zip(self.models, self.targets):
            target.load_state_dict(model.state_dict())
            target.eval()
        learning = BASELINE_LEARNING[name]
        self.learning_rate = learning["learning_rate"]
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=self.learning_rate)
                           for model in self.models]
        self.replay_capacity = learning["replay_capacity"]
        self.replays: List[Deque[tuple]] = [deque(maxlen=self.replay_capacity)
                                            for _ in range(self.agent_count)]
        self.batch_size = pilot_batch_size or learning["batch_size"]
        if update_every_slots < 1:
            raise ValueError("update_every_slots must be positive")
        self.update_every_slots = update_every_slots
        self.training_slot_count = 0
        if epsilon_decay_decisions is not None and epsilon_decay_decisions < 1:
            raise ValueError("epsilon_decay_decisions must be positive")
        self.epsilon_decay_decisions = epsilon_decay_decisions
        self.epsilon = learning["epsilon_start"] if training else 0.0
        self.epsilon_start = self.epsilon
        self.epsilon_end = learning["epsilon_end"]
        self.gamma = learning["gamma"]
        self.updates = [0] * self.agent_count
        self.losses: List[float] = []
        self.decisions = 0
        self.episode_reward_sum = 0.0
        self.episode_reward_count = 0
        self.pending: Dict[int, Tuple[int, torch.Tensor, int]] = {}
        self.path_rewards: Dict[int, float] = {}
        self.best_path_reward = -math.inf
        self.queue_history: List[Deque[int]] = [deque(maxlen=50)
                                                 for _ in range(config.num_satellites)]
        self.hidden = [torch.zeros(256) for _ in range(config.num_satellites)]
        self.graph_input_hidden = self.hidden

    def reset_runtime(self) -> None:
        """Start an independent evaluation scenario without carrying queue history."""
        self.pending.clear()
        self.path_rewards.clear()
        self.queue_history = [deque(maxlen=50) for _ in range(self.config.num_satellites)]
        self.hidden = [torch.zeros(256) for _ in range(self.config.num_satellites)]
        self.graph_input_hidden = self.hidden

    def _agent(self, node: int) -> int:
        return 0 if self.name == "dqn_ir_fixed_link" else node

    def _local_graph_features(self, node: int, snapshot: RoutingSnapshot) -> List[float]:
        plane, sat = divmod(node, self.config.satellites_per_plane)
        radius = self.config.earth_radius_km + self.config.orbital_altitude_km
        x, y, z = snapshot.positions[node]
        history = self.queue_history[node]
        average = sum(history) / len(history) if history else snapshot.queue_lengths[node]
        return [plane / (self.config.orbital_planes - 1),
                sat / (self.config.satellites_per_plane - 1),
                x / radius, y / radius, z / radius,
                snapshot.queue_lengths[node] / 100, average / 100]

    def _refresh_graph_hidden(self, snapshot: RoutingSnapshot) -> None:
        previous = self.hidden
        refreshed = []
        for node in range(self.config.num_satellites):
            self.queue_history[node].append(snapshot.queue_lengths[node])
        for node in range(self.config.num_satellites):
            slots = direction_slots(node, snapshot.neighbors[node], self.config)
            local = torch.tensor([self._local_graph_features(node, snapshot)], dtype=torch.float32,
                                 device=self.device)
            neighbors = torch.stack([previous[peer] if peer is not None else torch.zeros(256)
                                     for peer in slots]).unsqueeze(0).to(self.device)
            mask = torch.tensor([[float(peer is not None) for peer in slots]],
                                device=self.device)
            with torch.no_grad():
                refreshed.append(self.models[node].encode(local, neighbors, mask)[0].detach().cpu())
        self.graph_input_hidden = previous
        self.hidden = refreshed

    def _state(
        self, decision: PacketDecision, snapshot: RoutingSnapshot,
        hidden: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        node = decision.current
        destination = decision.destination
        slots = direction_slots(node, snapshot.neighbors[node], self.config)
        radius = self.config.earth_radius_km + self.config.orbital_altitude_km
        destination_position = snapshot.positions[destination]
        if self.name == "graphpr_adapted":
            source_hidden = hidden if hidden is not None else self.graph_input_hidden
            neighborhood = []
            for peer in slots:
                neighborhood.extend((source_hidden[peer] if peer is not None else torch.zeros(256)).tolist())
            destination_plane, destination_sat = divmod(destination, self.config.satellites_per_plane)
            packet_features = [destination_plane / (self.config.orbital_planes - 1),
                               destination_sat / (self.config.satellites_per_plane - 1),
                               *(coordinate / radius for coordinate in destination_position),
                               snapshot.queue_lengths[node] / 100]
            values = (self._local_graph_features(node, snapshot) + neighborhood +
                      [float(peer is not None) for peer in slots] + packet_features)
        elif self.name == "fdr_marl_adapted":
            values = [*(coordinate / radius for coordinate in destination_position),
                      snapshot.queue_lengths[node] / 100]
            for peer in slots:
                if peer is None:
                    values.extend([0.0] * 6)
                    continue
                relative = [(snapshot.positions[peer][index] - destination_position[index]) / (2 * radius)
                            for index in range(3)]
                erpd_ms = _distance(snapshot.positions[peer], destination_position) / 299_792.458 * 1000
                values.extend(relative + [snapshot.queue_lengths[peer] / 100,
                                          erpd_ms / 20.0, 1.0])
        else:
            maximum_grid = (self.config.orbital_planes // 2 +
                            self.config.satellites_per_plane // 2)
            tx_over_slot = (self.config.packet_size_bytes * 8 /
                            (self.config.isl_capacity_mbps * 1_000_000) /
                            self.config.slot_duration_s)
            values = []
            for peer in slots:
                if peer is None:
                    values.extend([0.0] * 5)
                else:
                    values.extend([snapshot.queue_lengths[peer] / 100,
                                   grid_manhattan_distance(peer, destination, self.config) / maximum_grid,
                                   float(peer == destination), tx_over_slot, 1.0])
        return torch.tensor(values, dtype=torch.float32)

    def route_packets(
        self, decisions: Sequence[PacketDecision], snapshot: RoutingSnapshot
    ) -> Dict[int, int]:
        if self.name == "graphpr_adapted":
            self._refresh_graph_hidden(snapshot)
        grouped: Dict[int, List[Tuple[PacketDecision, torch.Tensor]]] = {}
        for decision in decisions:
            agent = self._agent(decision.current)
            grouped.setdefault(agent, []).append((decision, self._state(decision, snapshot)))
        actions = {}
        for agent, group in grouped.items():
            states = torch.stack([state for _, state in group]).to(self.device)
            with torch.no_grad():
                q_values = self.models[agent](states).cpu()
            for row, (decision, state) in enumerate(group):
                slots = direction_slots(decision.current, decision.feasible_neighbors, self.config)
                feasible_indices = [index for index, peer in enumerate(slots) if peer is not None]
                if not feasible_indices:
                    raise ValueError("No feasible action for packet decision")
                if self.rng.random() < self.epsilon:
                    action_index = self.rng.choice(feasible_indices)
                else:
                    action_index = max(feasible_indices, key=lambda index: (float(q_values[row, index]), -index))
                actions[decision.packet_id] = slots[action_index]
                if self.training:
                    self.pending[decision.packet_id] = (agent, state, action_index)
                self.decisions += 1
        return actions

    def _reward(self, event: PacketTransitionEvent, before: RoutingSnapshot) -> float:
        decision = event.decision
        if self.name == "graphpr_adapted":
            if event.delivered:
                return 1.0
            if event.dropped:
                return -1.0
            rsph = residual_shortest_path_hops(
                decision.current, event.action, decision.destination,
                before.neighbors, self.config.max_hops,
            )
            propagation_ms = _distance(before.positions[decision.current],
                                       before.positions[event.action]) / 299_792.458 * 1000
            return 0.6 * rsph - 0.55 * event.queue_wait_slots - 0.45 * propagation_ms / 20
        if self.name == "fdr_marl_adapted":
            if event.delivered:
                return 50 * max(0, self.config.max_hops - decision.hops - 1) / 1000
            if event.dropped:
                return 0.0
            q = before.queue_lengths[event.action]
            if q >= 100:
                return 0.0
            coefficient = 0.5 if q <= 64 else 0.05
            erpd_ms = _distance(before.positions[event.action],
                                before.positions[decision.destination]) / 299_792.458 * 1000
            return (coefficient / (1 + q / 36) + 1000 / (1 + erpd_ms / 20)) / 1000
        if event.delivered:
            return 1.0
        if event.dropped:
            return -1.0
        q_ratio = before.queue_lengths[event.action] / 100
        grid_max = self.config.orbital_planes // 2 + self.config.satellites_per_plane // 2
        grid_ratio = grid_manhattan_distance(event.action, decision.destination, self.config) / grid_max
        hop_term = 1 / (1 + math.exp(-(decision.hops + 1 - 18)))
        tx_ratio = (self.config.packet_size_bytes * 8 /
                    (self.config.isl_capacity_mbps * 1_000_000) /
                    self.config.slot_duration_s)
        return -(0.4 * tx_ratio + 0.4 * q_ratio + 0.1 * grid_ratio + 0.1 * hop_term)

    def on_slot_end(
        self, events: Sequence[PacketTransitionEvent],
        before: RoutingSnapshot, after: RoutingSnapshot,
    ) -> None:
        if not self.training:
            return
        for event in events:
            agent, state, action_index = self.pending.pop(event.decision.packet_id)
            reward = self._reward(event, before)
            self.episode_reward_sum += reward
            self.episode_reward_count += 1
            done = event.delivered or event.dropped
            next_agent = self._agent(event.next_node)
            if done:
                next_state = torch.zeros_like(state)
                next_mask = (False, False, False, False)
            else:
                next_decision = PacketDecision(
                    packet_id=event.decision.packet_id, current=event.next_node,
                    destination=event.decision.destination, hops=event.decision.hops + 1,
                    slot=after.slot + 1,
                    feasible_neighbors=after.neighbors[event.next_node],
                )
                next_state = self._state(next_decision, after, hidden=self.hidden)
                next_mask = tuple(peer is not None for peer in direction_slots(
                    event.next_node, after.neighbors[event.next_node], self.config))
            self.replays[agent].append((state.to(torch.float16), action_index, reward,
                                        next_state.to(torch.float16),
                                        done, next_agent, next_mask))
            if self.name == "dqn_ir_fixed_link":
                total = self.path_rewards.get(event.decision.packet_id, 0.0) + reward
                if done:
                    self.path_rewards.pop(event.decision.packet_id, None)
                    if event.delivered and total > self.best_path_reward:
                        self.best_path_reward = total
                        self.targets[0].load_state_dict(self.models[0].state_dict())
                        self.updates[0] = 0
                else:
                    self.path_rewards[event.decision.packet_id] = total
        self.training_slot_count += 1
        if self.training_slot_count % self.update_every_slots == 0:
            for agent in range(self.agent_count):
                if len(self.replays[agent]) >= self.batch_size:
                    self._update(agent)

    def _update(self, agent: int) -> None:
        batch = self.rng.sample(self.replays[agent], self.batch_size)
        states = torch.stack([row[0] for row in batch]).to(self.device, dtype=torch.float32)
        action_indices = torch.tensor([row[1] for row in batch], dtype=torch.long,
                                      device=self.device)
        chosen = self.models[agent](states).gather(1, action_indices[:, None]).squeeze(1)
        targets = [float(row[2]) for row in batch]
        next_groups: Dict[int, List[int]] = {}
        for index, row in enumerate(batch):
            if not row[4] and any(row[6]):
                next_groups.setdefault(row[5], []).append(index)
        with torch.no_grad():
            for next_agent, indices in next_groups.items():
                next_states = torch.stack([batch[index][3] for index in indices]).to(
                    self.device, dtype=torch.float32
                )
                q_values = self.targets[next_agent](next_states).cpu()
                for row_index, batch_index in enumerate(indices):
                    feasible = batch[batch_index][6]
                    targets[batch_index] += self.gamma * max(
                        float(q_values[row_index, action])
                        for action, permitted in enumerate(feasible) if permitted
                    )
        target = torch.tensor(targets, dtype=torch.float32, device=self.device)
        loss = (F.smooth_l1_loss(chosen, target) if self.name == "graphpr_adapted"
                else F.mse_loss(chosen, target))
        self.optimizers[agent].zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.models[agent].parameters(), 10.0)
        self.optimizers[agent].step()
        self.losses.append(float(loss.item()))
        self.updates[agent] += 1
        if self.name != "graphpr_adapted" and self.updates[agent] % 500 == 0:
            self.targets[agent].load_state_dict(self.models[agent].state_dict())

    def finish_episode(self, total_train_decisions: Optional[int] = None) -> None:
        if self.name == "graphpr_adapted":
            for model, target in zip(self.models, self.targets):
                target.load_state_dict(model.state_dict())
        if self.epsilon_decay_decisions is not None:
            if total_train_decisions is None:
                raise ValueError("Decision-normalized epsilon requires cumulative train decisions")
            fraction = min(1.0, total_train_decisions / self.epsilon_decay_decisions)
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon_start * (self.epsilon_end / self.epsilon_start) ** fraction,
            )
        else:
            self.epsilon = max(self.epsilon_end, self.epsilon * 0.995)

    def checkpoint_state(self) -> dict:
        """Complete trainable state at an episode boundary, including replay and RNG."""
        if self.pending:
            raise RuntimeError("Cannot checkpoint with unresolved packet transitions")
        return {
            "name": self.name,
            "agent_count": self.agent_count,
            "batch_size": self.batch_size,
            "replay_capacity": self.replay_capacity,
            "update_every_slots": self.update_every_slots,
            "epsilon_decay_decisions": self.epsilon_decay_decisions,
            "training_slot_count": self.training_slot_count,
            "model_states": [model.state_dict() for model in self.models],
            "target_states": [target.state_dict() for target in self.targets],
            "optimizer_states": [optimizer.state_dict() for optimizer in self.optimizers],
            "replays": [list(replay) for replay in self.replays],
            "rng_state": self.rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
            "epsilon": self.epsilon,
            "updates": list(self.updates),
            "decisions": self.decisions,
            "best_path_reward": self.best_path_reward,
            "losses": list(self.losses),
            "episode_reward_sum": self.episode_reward_sum,
            "episode_reward_count": self.episode_reward_count,
        }

    def load_checkpoint_state(self, saved: dict) -> None:
        """Restore exact optimizer/replay/exploration state on this policy's device."""
        for key in ("name", "agent_count", "batch_size", "replay_capacity",
                    "update_every_slots", "epsilon_decay_decisions"):
            value = saved.get(key) if key == "epsilon_decay_decisions" else saved[key]
            if value != getattr(self, key):
                raise ValueError(f"Checkpoint {key} mismatch: {value} != {getattr(self, key)}")
        if any(len(saved[key]) != self.agent_count for key in
               ("model_states", "target_states", "optimizer_states", "replays", "updates")):
            raise ValueError("Checkpoint agent count is inconsistent")
        for model, state in zip(self.models, saved["model_states"]):
            model.load_state_dict(state)
        for target, state in zip(self.targets, saved["target_states"]):
            target.load_state_dict(state)
        for optimizer, state in zip(self.optimizers, saved["optimizer_states"]):
            optimizer.load_state_dict(state)
        self.replays = [deque(rows, maxlen=self.replay_capacity) for rows in saved["replays"]]
        self.rng.setstate(saved["rng_state"])
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        if self.device.type == "cuda" and saved["cuda_rng_states"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng_states"]])
        self.epsilon = float(saved["epsilon"])
        self.updates = list(saved["updates"])
        self.decisions = int(saved["decisions"])
        self.training_slot_count = int(saved["training_slot_count"])
        self.best_path_reward = float(saved["best_path_reward"])
        self.losses = list(saved["losses"])
        self.episode_reward_sum = float(saved["episode_reward_sum"])
        self.episode_reward_count = int(saved["episode_reward_count"])
        self.reset_runtime()

    def parameter_count(self) -> int:
        return sum(sum(parameter.numel() for parameter in model.parameters())
                   for model in self.models)
