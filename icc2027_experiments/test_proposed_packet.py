"""Conservation, state mapping, and resume tests for per-packet Proposed."""

import tempfile
import unittest
from pathlib import Path

import torch

from .baseline_packet import PacketDecision, PerPacketRoutingEnvironment, RoutingSnapshot
from .baseline_train import train
from .config import SimulationConfig
from .environment import LEORoutingEnvironment
from .proposed_packet import PACKET_ABLATION_VARIANTS, ProposedPacketPolicy


class ProposedPacketTests(unittest.TestCase):
    def setUp(self):
        (Path.cwd() / "outputs").mkdir(exist_ok=True)

    def test_packet_ablation_keeps_state_reward_and_feasible_neighbors(self):
        config = SimulationConfig()
        environment = LEORoutingEnvironment(config, 20.0)
        snapshot = RoutingSnapshot(
            0, tuple(environment.positions(0)),
            tuple(len(queue) for queue in environment.node_queues),
            tuple(tuple(row) for row in environment.neighbors_at(0)),
        )
        full = ProposedPacketPolicy(config, 11, training=False)
        decision = PacketDecision(1, 0, 10, 0, 0, snapshot.neighbors[0])
        full_state = full._state(0, 10, snapshot)
        full_parameters = full.parameter_count()
        for method, variant in PACKET_ABLATION_VARIANTS.items():
            with self.subTest(method=method):
                policy = ProposedPacketPolicy(config, 11, training=False,
                                              variant=variant)
                self.assertEqual(policy.name, method)
                self.assertEqual(policy.epsilon, 0.0)
                self.assertEqual(policy.observation_window, full.observation_window)
                self.assertEqual(policy.learning, full.learning)
                self.assertTrue(torch.equal(policy._state(0, 10, snapshot), full_state))
                self.assertLess(policy.parameter_count(), full_parameters)
                result = policy.route_packets([decision], snapshot)
                self.assertIn(result[1], decision.feasible_neighbors)

    def test_packet_ablation_training_resume(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "outputs") as directory:
            root = Path(directory)
            common = dict(method="proposed_dqn_packet", seed=11, load_mbps=20.0,
                          episodes=2, warmup_slots=1, train_slots=2, batch_size=2)
            train(output=root / "full", **common)
            train(output=root / "split", stop_after_episode=1, **common)
            train(output=root / "split", resume=True, **common)
            full = torch.load(root / "full" / "checkpoint_latest.pt", map_location="cpu")
            resumed = torch.load(root / "split" / "checkpoint_latest.pt", map_location="cpu")
            self.assertEqual(full["signature"]["method"], "proposed_dqn_packet")
            self.assertEqual(full["policy"]["name"], "proposed_dqn_packet")
            self.assertEqual(full["rows"][-1]["total_train_decisions"],
                             resumed["rows"][-1]["total_train_decisions"])
            for key in full["policy"]["model_state"]:
                self.assertTrue(torch.equal(full["policy"]["model_state"][key],
                                            resumed["policy"]["model_state"][key]))

    def test_six_feature_state_matches_original_proposed_environment(self):
        config = SimulationConfig()
        env = LEORoutingEnvironment(config, 20.0)
        policy = ProposedPacketPolicy(config, 11)
        snapshot = RoutingSnapshot(
            0, tuple(env.positions(0)), tuple(len(queue) for queue in env.node_queues),
            tuple(tuple(row) for row in env.neighbors_at(0)),
        )
        self.assertTrue(torch.equal(policy._state(0, 10, snapshot),
                                    env.state_tensor(0, 10, 0)))

    def test_all_packet_decisions_enter_training_and_conserve_packets(self):
        config = SimulationConfig()
        policy = ProposedPacketPolicy(config, 11, batch_size=2)
        trace = (((0, 10), (0, 10), (1, 20)),)
        env = PerPacketRoutingEnvironment(config, 20.0, trace, policy)
        env.advance_background()
        self.assertEqual(policy.decisions, 3)
        self.assertEqual(policy.episode_reward_count, 3)
        self.assertEqual(len(policy.replays[0]), 3)
        self.assertEqual(env.background_generated,
                         env.background_delivered + env.background_dropped +
                         sum(len(queue) for queue in env.node_queues))

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "outputs") as directory:
            root = Path(directory)
            common = dict(method="proposed_gat_lstm_packet", seed=11, load_mbps=20.0,
                          episodes=2, warmup_slots=1, train_slots=2, batch_size=2)
            train(output=root / "full", **common)
            train(output=root / "split", stop_after_episode=1, **common)
            train(output=root / "split", resume=True, **common)
            with (root / "full" / "checkpoint_latest.pt").open("rb") as stream:
                full = torch.load(stream, map_location="cpu")
            with (root / "split" / "checkpoint_latest.pt").open("rb") as stream:
                resumed = torch.load(stream, map_location="cpu")
            self.assertEqual(full["next_episode"], 2)
            self.assertEqual(full["policy"]["decisions"], resumed["policy"]["decisions"])
            self.assertEqual(full["policy"]["rng_state"], resumed["policy"]["rng_state"])
            for first, second in zip(full["rows"], resumed["rows"]):
                self.assertEqual({key: value for key, value in first.items() if key != "wall_seconds"},
                                 {key: value for key, value in second.items() if key != "wall_seconds"})
            for key in full["policy"]["model_state"]:
                self.assertTrue(torch.equal(full["policy"]["model_state"][key],
                                            resumed["policy"]["model_state"][key]))


if __name__ == "__main__":
    unittest.main()
