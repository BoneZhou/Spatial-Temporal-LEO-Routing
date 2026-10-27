"""Per-packet training adapter for the original Proposed GAT-LSTM-DQN network.

The network, six-feature state, reward coefficients, and published project
learning settings are retained.  All packets now contribute transitions on
the same exogenous traffic trace as the adapted external baselines.
"""

import math
import random
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .baseline_packet import PacketDecision, PacketTransitionEvent, RoutingSnapshot
from .config import LearningConfig, SimulationConfig
from .models import VARIANTS, UnifiedRoutingQNetwork


PACKET_ABLATION_VARIANTS = {
    "proposed_dqn_packet": "dqn",
    "proposed_lstm_dqn_packet": "lstm_dqn",
    "proposed_gat_dqn_packet": "gat_dqn",
}


class ProposedPacketPolicy:
    name = "proposed_gat_lstm_packet"

    def __init__(
        self, config: SimulationConfig, seed: int, *, training: bool = True,
        batch_size: Optional[int] = None, device: str = "cpu", observation_window: int = 5,
        update_every_slots: int = 1,
        epsilon_decay_decisions: Optional[int] = None,
        variant: str = "gat_lstm_dqn",
    ):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown Proposed variant: {variant}")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if observation_window < 1 or update_every_slots < 1:
            raise ValueError("Invalid observation window or update interval")
        self.config = config
        self.variant = variant
        self.name = (ProposedPacketPolicy.name if variant == "gat_lstm_dqn" else
                     f"proposed_{variant}_packet")
        self.learning = LearningConfig()
        self.device = torch.device(device)
        self.observation_window = observation_window
        self.update_every_slots = update_every_slots
        if epsilon_decay_decisions is not None and epsilon_decay_decisions < 1:
            raise ValueError("epsilon_decay_decisions must be positive")
        self.epsilon_decay_decisions = epsilon_decay_decisions
        self.training_slot_count = 0
        self.training = training
        self.rng = random.Random(seed + 9001)
        torch.manual_seed(seed)
        self.model = UnifiedRoutingQNetwork(
            variant, 6, self.learning.hidden_dim, self.learning.lstm_hidden_dim
        ).to(self.device)
        self.target = UnifiedRoutingQNetwork(
            variant, 6, self.learning.hidden_dim, self.learning.lstm_hidden_dim
        ).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning.learning_rate)
        self.agent_count = 1
        self.batch_size = batch_size or self.learning.batch_size
        self.replay_capacity = self.learning.replay_buffer_size
        self.replays: List[Deque[tuple]] = [deque(maxlen=self.replay_capacity)]
        self.epsilon = self.learning.epsilon_start if training else 0.0
        self.epsilon_start = self.epsilon
        self.epsilon_end = self.learning.epsilon_end
        self.updates = [0]
        self.losses: List[float] = []
        self.decisions = 0
        self.episode_reward_sum = 0.0
        self.episode_reward_count = 0
        self.histories: Dict[int, Deque[torch.Tensor]] = {}
        self.pending: Dict[int, Tuple[torch.Tensor, int, int]] = {}
        self.adjacencies: Dict[int, torch.Tensor] = {}

    def reset_runtime(self) -> None:
        self.histories.clear()
        self.pending.clear()

    def _adjacency(self, snapshot: RoutingSnapshot) -> torch.Tensor:
        if snapshot.slot in self.adjacencies:
            return self.adjacencies[snapshot.slot]
        matrix = torch.eye(self.config.num_satellites, dtype=torch.bool)
        for node, peers in enumerate(snapshot.neighbors):
            matrix[node, list(peers)] = True
        self.adjacencies[snapshot.slot] = matrix
        return matrix

    def _state(self, current: int, destination: int, snapshot: RoutingSnapshot) -> torch.Tensor:
        """Exactly the original six node features, from an immutable slot snapshot."""
        scale = 2 * (self.config.earth_radius_km + self.config.orbital_altitude_km)
        target = snapshot.positions[destination]
        features = []
        for node, position in enumerate(snapshot.positions):
            delta = [position[index] - target[index] for index in range(3)]
            distance = math.sqrt(sum(value * value for value in delta))
            features.append([value / scale for value in delta] + [
                snapshot.queue_lengths[node] / self.config.max_queue_packets,
                distance / scale, float(node == current),
            ])
        return torch.tensor(features, dtype=torch.float32)

    def route_packets(
        self, decisions: Sequence[PacketDecision], snapshot: RoutingSnapshot
    ) -> Dict[int, int]:
        if not decisions:
            return {}
        state_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        sequences = []
        for decision in decisions:
            key = (decision.current, decision.destination)
            if key not in state_cache:
                state_cache[key] = self._state(*key, snapshot)
            state = state_cache[key]
            history = self.histories.get(decision.packet_id)
            if history is None:
                history = deque([state] * self.observation_window,
                                maxlen=self.observation_window)
                self.histories[decision.packet_id] = history
            else:
                history[-1] = state
            sequences.append(torch.stack(tuple(history)))
        adjacency = self._adjacency(snapshot).to(self.device)
        q_batches = []
        with torch.no_grad():
            for start in range(0, len(sequences), 64):
                batch = torch.stack(sequences[start:start + 64]).to(self.device)
                q_batches.append(self.model(batch, adjacency).cpu())
        q_values = torch.cat(q_batches)
        actions = {}
        for index, decision in enumerate(decisions):
            feasible = decision.feasible_neighbors
            if self.rng.random() < self.epsilon:
                action = self.rng.choice(feasible)
            else:
                action = max(feasible, key=lambda node: (float(q_values[index, node]), -node))
            actions[decision.packet_id] = action
            self.pending[decision.packet_id] = (sequences[index], action, snapshot.slot)
            self.decisions += 1
        return actions

    def _reward(self, event: PacketTransitionEvent, before: RoutingSnapshot) -> float:
        if event.delivered:
            return 10.0
        if event.dropped:
            return -10.0
        current = event.decision.current
        destination = event.decision.destination
        action = event.action
        positions = before.positions
        old_distance = math.dist(positions[current], positions[destination])
        new_distance = math.dist(positions[action], positions[destination])
        distance_reward = 1.0 if new_distance < old_distance else -1.0
        congestion = 2.0 * min(1.0, before.queue_lengths[current] /
                               self.config.max_queue_packets)
        return distance_reward - congestion - 0.1

    def on_slot_end(
        self, events: Sequence[PacketTransitionEvent],
        before: RoutingSnapshot, after: RoutingSnapshot,
    ) -> None:
        next_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._adjacency(after)
        for event in events:
            packet_id = event.decision.packet_id
            sequence, action, before_slot = self.pending.pop(packet_id)
            done = event.delivered or event.dropped
            if done:
                self.histories.pop(packet_id, None)
                next_sequence = torch.zeros_like(sequence)
                next_mask = torch.zeros(self.config.num_satellites, dtype=torch.bool)
            else:
                key = (event.next_node, event.decision.destination)
                if key not in next_cache:
                    next_cache[key] = self._state(*key, after)
                state = next_cache[key]
                history = self.histories[packet_id]
                history.append(state)
                next_sequence = torch.stack(tuple(history))
                next_mask = torch.zeros(self.config.num_satellites, dtype=torch.bool)
                next_mask[list(after.neighbors[event.next_node])] = True
            if self.training:
                reward = self._reward(event, before)
                self.episode_reward_sum += reward
                self.episode_reward_count += 1
                self.replays[0].append((
                    sequence.to(torch.float16), action, reward,
                    next_sequence.to(torch.float16), done,
                    before_slot, after.slot, next_mask,
                ))
        if self.training:
            self.training_slot_count += 1
            if (self.training_slot_count % self.update_every_slots == 0 and
                    len(self.replays[0]) >= self.batch_size):
                self._update()

    def _update(self) -> None:
        batch = self.rng.sample(self.replays[0], self.batch_size)
        states = torch.stack([row[0] for row in batch]).to(self.device, dtype=torch.float32)
        actions = torch.tensor([row[1] for row in batch], dtype=torch.long,
                               device=self.device)
        adjacency = torch.stack([self.adjacencies[row[5]] for row in batch]).to(self.device)
        selected = self.model(states, adjacency).gather(1, actions[:, None]).squeeze(1)
        targets = torch.tensor([row[2] for row in batch], dtype=torch.float32,
                               device=self.device)
        active = [index for index, row in enumerate(batch) if not row[4] and row[7].any()]
        with torch.no_grad():
            for start in range(0, len(active), 64):
                indices = active[start:start + 64]
                next_states = torch.stack([batch[index][3] for index in indices]).to(
                    self.device, dtype=torch.float32
                )
                next_adjacency = torch.stack([self.adjacencies[batch[index][6]]
                                              for index in indices]).to(
                    self.device
                )
                q = self.target(next_states, next_adjacency)
                masks = torch.stack([batch[index][7] for index in indices]).to(self.device)
                next_value = q.masked_fill(~masks, -1e9).max(dim=1).values
                targets[indices] = targets[indices] + self.learning.discount_factor * next_value
        loss = nn.functional.smooth_l1_loss(selected, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.losses.append(float(loss.item()))
        self.updates[0] += 1
        if self.updates[0] % self.learning.target_update_steps == 0:
            self.target.load_state_dict(self.model.state_dict())

    def finish_episode(self, total_train_decisions: Optional[int] = None) -> None:
        if self.epsilon_decay_decisions is not None:
            if total_train_decisions is None:
                raise ValueError("Decision-normalized epsilon requires cumulative train decisions")
            fraction = min(1.0, total_train_decisions / self.epsilon_decay_decisions)
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon_start * (self.epsilon_end / self.epsilon_start) ** fraction,
            )
        else:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.learning.epsilon_decay)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def checkpoint_state(self) -> dict:
        if self.pending:
            raise RuntimeError("Cannot checkpoint with pending transitions")
        return {
            "name": self.name, "batch_size": self.batch_size,
            "replay_capacity": self.replay_capacity,
            "observation_window": self.observation_window,
            "update_every_slots": self.update_every_slots,
            "epsilon_decay_decisions": self.epsilon_decay_decisions,
            "model_state": self.model.state_dict(),
            "target_state": self.target.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "replay": list(self.replays[0]), "rng_state": self.rng.getstate(),
            "adjacencies": dict(self.adjacencies),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
            "epsilon": self.epsilon, "updates": list(self.updates),
            "decisions": self.decisions, "training_slot_count": self.training_slot_count,
            "losses": list(self.losses), "episode_reward_sum": self.episode_reward_sum,
            "episode_reward_count": self.episode_reward_count,
        }

    def load_checkpoint_state(self, saved: dict) -> None:
        for key in ("name", "batch_size", "replay_capacity", "observation_window",
                    "update_every_slots", "epsilon_decay_decisions"):
            value = saved.get(key) if key == "epsilon_decay_decisions" else saved[key]
            if value != getattr(self, key):
                raise ValueError(f"Checkpoint {key} mismatch")
        self.model.load_state_dict(saved["model_state"])
        self.target.load_state_dict(saved["target_state"])
        self.optimizer.load_state_dict(saved["optimizer_state"])
        self.replays = [deque(saved["replay"], maxlen=self.replay_capacity)]
        self.adjacencies = dict(saved["adjacencies"])
        self.rng.setstate(saved["rng_state"])
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        if self.device.type == "cuda" and saved["cuda_rng_states"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng_states"]])
        self.epsilon = float(saved["epsilon"])
        self.updates = list(saved["updates"])
        self.decisions = int(saved["decisions"])
        self.training_slot_count = int(saved["training_slot_count"])
        self.losses = list(saved["losses"])
        self.episode_reward_sum = float(saved["episode_reward_sum"])
        self.episode_reward_count = int(saved["episode_reward_count"])
        self.reset_runtime()
