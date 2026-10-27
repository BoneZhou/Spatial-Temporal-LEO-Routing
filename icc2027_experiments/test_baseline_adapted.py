import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from .baseline_methods import (
    BASELINE_NAMES, DQNIRResNet, FDRQ, GraphPRQ, AdaptedBaselinePolicy,
    grid_manhattan_distance, residual_shortest_path_hops,
)
from .baseline_packet import (
    PacketDecision, PacketTransitionEvent, PerPacketRoutingEnvironment,
    RoutingSnapshot, direction_slots, make_training_trace,
)
from .config import SimulationConfig
from .environment import LEORoutingEnvironment
from .baseline_train import train
from .packet_validation import validate_evaluation_metrics


class FirstNeighborRouter:
    def __init__(self):
        self.seen = []
        self.events = []

    def route_packets(self, decisions, snapshot):
        self.seen.extend(decisions)
        return {decision.packet_id: decision.feasible_neighbors[0] for decision in decisions}

    def on_slot_end(self, events, before, after):
        self.events.extend(events)


class AdaptedBaselineTests(unittest.TestCase):
    def setUp(self):
        (Path.cwd() / "outputs").mkdir(exist_ok=True)
        self.config = SimulationConfig()

    def test_action_slots_cover_actual_dynamic_isls(self):
        env = LEORoutingEnvironment(self.config, 180.0)
        for step in (0, 10, 1000):
            for node, neighbors in enumerate(env.neighbors_at(step)):
                slots = direction_slots(node, neighbors, self.config)
                self.assertEqual(set(peer for peer in slots if peer is not None), set(neighbors))
                self.assertEqual(len(slots), 4)

    def test_shared_queue_service_and_independent_packet_ids(self):
        router = FirstNeighborRouter()
        trace = (tuple((0, 1) for _ in range(100)),)
        env = PerPacketRoutingEnvironment(self.config, 180.0, trace, router)
        env.advance_background()
        self.assertEqual(env.background_generated, 100)
        self.assertEqual(env.route_decisions, 36)
        self.assertEqual(len(router.seen), 36)
        self.assertEqual(len({item.packet_id for item in router.seen}), 36)
        self.assertEqual(len(router.events), 36)
        self.assertEqual(env.background_dropped, 0)
        self.assertTrue(all(len(queue) <= 100 for queue in env.node_queues))

    def test_trace_reproducible_and_aggregate_load(self):
        first = make_training_trace(self.config, 180, 1100017, 5)
        second = make_training_trace(self.config, 180, 1100017, 5)
        self.assertEqual(first, second)
        self.assertGreater(sum(len(slot) for slot in first), 0)
        self.assertEqual(sum(len(slot) for slot in make_training_trace(self.config, 0, 11, 2)), 0)

    def test_rsph_and_grid_distance(self):
        graph = ((1, 2), (0, 3), (0, 3), (1, 2))
        self.assertEqual(residual_shortest_path_hops(0, 1, 3, graph), 1)
        disconnected = ((1,), (0,), ())
        self.assertEqual(residual_shortest_path_hops(0, 1, 2, disconnected), -30)
        self.assertEqual(grid_manhattan_distance(0, 44, self.config), 2)
        self.assertEqual(max(grid_manhattan_distance(i, j, self.config)
                             for i in range(45) for j in range(45)), 8)

    def test_fdr_thresholds_and_empty_queue_reward_are_finite(self):
        policy = AdaptedBaselinePolicy("fdr_marl_adapted", self.config, 11, pilot_batch_size=2)
        env = LEORoutingEnvironment(self.config, 180.0)
        neighbors = tuple(tuple(row) for row in env.neighbors_at(0))
        positions = tuple(env.positions(0))
        decision = PacketDecision(1, 0, 1, 0, 0, neighbors[0])
        event = PacketTransitionEvent(decision, 14, 0, False, False, 14)

        def reward_at(queue_length):
            queues = [0] * 45
            queues[14] = queue_length
            snapshot = RoutingSnapshot(0, positions, tuple(queues), neighbors)
            return policy._reward(event, snapshot)

        self.assertGreater(reward_at(64), reward_at(65))
        self.assertEqual(reward_at(100), 0.0)
        self.assertTrue(math.isfinite(reward_at(0)))

    def test_validation_delay_is_undefined_only_when_nothing_is_delivered(self):
        result = {
            "method": "dqn_ir_fixed_link",
            "traffic_seed": 601,
            "throughput_mbps": 0.0,
            "packet_loss_rate": 0.95,
            "average_queue_packets": 10.0,
            "delivered_packets": 0,
            "end_to_end_delay_ms": None,
        }
        validate_evaluation_metrics(result)

        result["delivered_packets"] = 1
        with self.assertRaisesRegex(RuntimeError, "end_to_end_delay_ms"):
            validate_evaluation_metrics(result)

        result["delivered_packets"] = 0
        result["end_to_end_delay_ms"] = 0.0
        with self.assertRaisesRegex(RuntimeError, "undefined delay"):
            validate_evaluation_metrics(result)

        result["end_to_end_delay_ms"] = None
        result["throughput_mbps"] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "throughput_mbps"):
            validate_evaluation_metrics(result)

    def test_dqn_fixed_link_term_does_not_fabricate_snr(self):
        policy = AdaptedBaselinePolicy("dqn_ir_fixed_link", self.config, 11, pilot_batch_size=2)
        env = LEORoutingEnvironment(self.config, 180.0)
        snapshot = RoutingSnapshot(
            0, tuple(env.positions(0)), tuple([0] * 45),
            tuple(tuple(row) for row in env.neighbors_at(0)),
        )
        decision = PacketDecision(1, 0, 10, 0, 0, snapshot.neighbors[0])
        state = policy._state(decision, snapshot)
        for index in range(4):
            if state[index * 5 + 4] == 1:
                self.assertAlmostEqual(float(state[index * 5 + 3]), 0.002)

    def test_packet_id_survives_hop_and_ttl_drop(self):
        config = replace(self.config, max_hops=1)
        router = FirstNeighborRouter()
        env = PerPacketRoutingEnvironment(config, 180.0, (((0, 10),), ()), router)
        env.advance_background()
        self.assertTrue(any(packet[4] == 1 for queue in env.node_queues for packet in queue))
        env.advance_background()
        self.assertEqual([item.packet_id for item in router.seen], [1])
        self.assertEqual(env.background_ttl_dropped, 1)
        self.assertEqual(env.background_generated, env.background_dropped)

    def test_three_architectures_and_finite_rewards(self):
        self.assertEqual(tuple(GraphPRQ()(torch.zeros(1, 1041)).shape), (1, 4))
        self.assertEqual(tuple(FDRQ()(torch.zeros(1, 28)).shape), (1, 4))
        self.assertEqual(tuple(DQNIRResNet()(torch.zeros(1, 20)).shape), (1, 4))
        for name in BASELINE_NAMES:
            policy = AdaptedBaselinePolicy(name, self.config, 11, pilot_batch_size=2)
            router = policy
            trace = (((0, 1), (0, 1), (2, 3)),)
            env = PerPacketRoutingEnvironment(self.config, 180.0, trace, router)
            env.advance_background()
            self.assertEqual(env.route_decisions, 3)
            self.assertEqual(policy.decisions, 3)
            self.assertTrue(all(math.isfinite(loss) for loss in policy.losses))
            self.assertTrue(all(len(replay) <= policy.replay_capacity for replay in policy.replays))
            policy.finish_episode()

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "outputs") as directory:
            root = Path(directory)
            common = dict(method="dqn_ir_fixed_link", seed=11, load_mbps=180.0,
                          episodes=2, warmup_slots=1, train_slots=2, batch_size=2)
            train(output=root / "full", **common)
            train(output=root / "split", stop_after_episode=1, **common)
            train(output=root / "split", resume=True, **common)
            with (root / "full" / "checkpoint_latest.pt").open("rb") as stream:
                full = torch.load(stream, map_location="cpu")
            with (root / "split" / "checkpoint_latest.pt").open("rb") as stream:
                resumed = torch.load(stream, map_location="cpu")
            self.assertEqual(full["next_episode"], 2)
            self.assertEqual(resumed["next_episode"], 2)
            self.assertTrue((root / "split" / "training_complete.ok").exists())
            for first, second in zip(full["rows"], resumed["rows"]):
                self.assertEqual({key: value for key, value in first.items() if key != "wall_seconds"},
                                 {key: value for key, value in second.items() if key != "wall_seconds"})
            for key in full["policy"]["model_states"][0]:
                self.assertTrue(torch.equal(full["policy"]["model_states"][0][key],
                                            resumed["policy"]["model_states"][0][key]))
            self.assertEqual(full["policy"]["rng_state"], resumed["policy"]["rng_state"])
            self.assertEqual(full["policy"]["decisions"], resumed["policy"]["decisions"])

    def test_checkpoint_rejects_changed_training_signature(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "outputs") as directory:
            output = Path(directory) / "train"
            train("dqn_ir_fixed_link", 11, 180.0, 1, output,
                  warmup_slots=0, train_slots=1, batch_size=2)
            with self.assertRaises(ValueError):
                train("dqn_ir_fixed_link", 11, 240.0, 2, output,
                      warmup_slots=0, train_slots=1, batch_size=2, resume=True)

    def test_common_decision_budget_and_normalized_epsilon(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "outputs") as directory:
            output = Path(directory) / "train"
            result = train("dqn_ir_fixed_link", 11, 180.0, 10, output,
                           warmup_slots=1, train_slots=2, batch_size=2,
                           max_train_decisions=5000, epsilon_decay_decisions=1000,
                           checkpoint_every=3)
            self.assertTrue(result["budget_reached"])
            self.assertEqual(result["completed_episodes"], 2)
            self.assertLessEqual(result["total_train_decisions"], 5000)
            self.assertTrue((output / "training_complete.ok").exists())
            with (output / "checkpoint_latest.pt").open("rb") as stream:
                saved = torch.load(stream, map_location="cpu")
            self.assertEqual(saved["next_episode"], 2)
            self.assertAlmostEqual(saved["policy"]["epsilon"], 0.001)


if __name__ == "__main__":
    unittest.main()
