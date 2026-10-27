import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from .config import build_config
from .environment import LEORoutingEnvironment
from .models import VARIANTS, UnifiedRoutingQNetwork
from .run import (
    CUDA_BACKGROUND_BATCH_SIZE,
    evaluate_variant,
    resolve_variant_device,
    train_variant,
)


class UnifiedExperimentContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = build_config(11, 180.0, 5, "smoke")
        self.env = LEORoutingEnvironment(self.config.simulation, self.config.offered_load_mbps)
        self.env.reset(self.config.traffic_seed)

    def test_state_and_topology_dimensions(self) -> None:
        state = self.env.state_tensor(0, 1, 0)
        adjacency = self.env.adjacency_tensor(0)
        self.assertEqual(tuple(state.shape), (45, 6))
        self.assertEqual(tuple(adjacency.shape), (45, 45))
        self.assertTrue(torch.all(torch.diag(adjacency)))

    def test_dynamic_topology_contract(self) -> None:
        cfg = self.config.simulation
        period_slots = cfg.orbital_period_slots
        sampled_steps = [round(index * period_slots / 300) for index in range(300)]
        for step in sampled_steps:
            positions = self.env.positions(step)
            neighbors = self.env.neighbors_at(step)
            self.assertEqual(neighbors, self.env.neighbors_at(step))

            for node, peers in enumerate(neighbors):
                intra_plane = [peer for peer in peers if peer // 15 == node // 15]
                inter_plane = [peer for peer in peers if peer // 15 != node // 15]
                self.assertEqual(len(intra_plane), 2)
                self.assertLessEqual(len(inter_plane), 2)
                self.assertLessEqual(len(peers), 4)
                for peer in peers:
                    self.assertIn(node, neighbors[peer])
                    self.assertTrue(
                        self.env.has_line_of_sight(positions[node], positions[peer])
                    )
                for peer in inter_plane:
                    for endpoint in (node, peer):
                        x, y, z = positions[endpoint]
                        latitude = math.degrees(
                            math.asin(z / math.sqrt(x * x + y * y + z * z))
                        )
                        self.assertLessEqual(
                            abs(latitude), cfg.inter_plane_latitude_limit_deg + 1e-9
                        )

            reached = {0}
            frontier = [0]
            while frontier:
                node = frontier.pop()
                for peer in neighbors[node]:
                    if peer not in reached:
                        reached.add(peer)
                        frontier.append(peer)
            self.assertEqual(len(reached), cfg.num_satellites)

    def test_physical_orbital_time_scale(self) -> None:
        cfg = self.config.simulation
        self.assertAlmostEqual(cfg.orbital_period_s / 60.0, 95.9164, places=3)
        self.assertAlmostEqual(cfg.orbital_period_slots, 287_749.1513, places=3)

        initial = self.env.positions(0)
        one_orbit = self.env.positions(round(cfg.orbital_period_slots))
        maximum_return_error_km = max(
            self.env.distance(start, end)
            for start, end in zip(initial, one_orbit)
        )
        self.assertLess(maximum_return_error_km, 0.2)

        interval_config = replace(cfg, topology_update_interval_steps=5)
        interval_env = LEORoutingEnvironment(interval_config, 180.0)
        self.assertIs(interval_env.neighbors_at(0), interval_env.neighbors_at(4))
        interval_env.advance_background()
        self.assertEqual(interval_env.step_index, 1)

    def test_pilot_preset_is_non_formal_and_bounded(self) -> None:
        pilot = build_config(11, 180.0, 5, "pilot")
        self.assertEqual(pilot.mode, "pilot")
        self.assertEqual(pilot.learning.training_episodes, 5)
        self.assertEqual(pilot.learning.evaluation_episodes, 3)
        self.assertEqual(pilot.simulation.evaluation_warmup_slots, 3)
        self.assertEqual(pilot.simulation.evaluation_measurement_slots, 20)
        self.assertEqual(
            pilot.simulation.training_warmup_slots,
            pilot.simulation.evaluation_warmup_slots,
        )

    def test_all_ablation_outputs_have_common_action_shape(self) -> None:
        state = self.env.state_tensor(0, 1, 0)
        sequence = torch.stack([state] * self.config.observation_window).unsqueeze(0)
        adjacency = self.env.adjacency_tensor(0)
        for variant in VARIANTS:
            model = UnifiedRoutingQNetwork(variant, 6, 24, 24)
            output = model(sequence, adjacency)
            self.assertEqual(tuple(output.shape), (1, 45), variant)

    def test_gpu_selection_is_limited_to_gat_lstm(self) -> None:
        for variant in ("dqn", "lstm_dqn", "gat_dqn"):
            self.assertEqual(resolve_variant_device(variant, "cuda").type, "cpu")
        self.assertEqual(resolve_variant_device("gat_lstm_dqn", "cpu").type, "cpu")

        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(
                resolve_variant_device("gat_lstm_dqn", "auto").type,
                "cpu",
            )
            with self.assertRaisesRegex(RuntimeError, "cannot access CUDA"):
                resolve_variant_device("gat_lstm_dqn", "cuda")

        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(
                resolve_variant_device("gat_lstm_dqn", "auto").type,
                "cuda",
            )

    def test_explicit_gat_device_applies_to_both_gat_variants(self) -> None:
        for variant in ("dqn", "lstm_dqn"):
            self.assertEqual(
                resolve_variant_device(variant, "cpu", "cuda").type,
                "cpu",
            )
        with mock.patch("torch.cuda.is_available", return_value=True):
            for variant in ("gat_dqn", "gat_lstm_dqn"):
                self.assertEqual(
                    resolve_variant_device(variant, "cpu", "cuda").type,
                    "cuda",
                )

    def test_explicit_generic_device_applies_to_every_variant(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=True):
            for variant in VARIANTS:
                self.assertEqual(
                    resolve_variant_device(variant, "cpu", None, "cuda").type,
                    "cuda",
                )
        with mock.patch("torch.cuda.is_available", return_value=False):
            for variant in VARIANTS:
                self.assertEqual(
                    resolve_variant_device(variant, "cuda", "cuda", "auto").type,
                    "cpu",
                )
            with self.assertRaisesRegex(RuntimeError, "cannot access CUDA"):
                resolve_variant_device("dqn", "cpu", None, "cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_every_variant_runs_equivalent_forward_pass_on_cuda(self) -> None:
        torch.manual_seed(11)
        state = self.env.state_tensor(0, 1, 0)
        sequence = torch.stack([state] * self.config.observation_window).unsqueeze(0)
        adjacency = self.env.adjacency_tensor(0)
        for variant in VARIANTS:
            cpu_model = UnifiedRoutingQNetwork(variant, 6, 24, 24).eval()
            gpu_model = UnifiedRoutingQNetwork(variant, 6, 24, 24).cuda().eval()
            gpu_model.load_state_dict(cpu_model.state_dict())
            with torch.no_grad():
                cpu_output = cpu_model(sequence, adjacency)
                gpu_output = gpu_model(sequence.cuda(), adjacency.cuda()).cpu()
            self.assertTrue(
                torch.allclose(cpu_output, gpu_output, rtol=1e-4, atol=1e-5),
                variant,
            )

    def test_traffic_is_reproducible_by_seed(self) -> None:
        first = LEORoutingEnvironment(self.config.simulation, 180.0)
        second = LEORoutingEnvironment(self.config.simulation, 180.0)
        first.reset(1234)
        second.reset(1234)
        self.assertEqual(first.sample_source_destination(), second.sample_source_destination())
        first.reset_measurement_counters()
        second.reset_measurement_counters()
        first.advance_background()
        second.advance_background()
        self.assertEqual(first.background_generated, second.background_generated)
        self.assertEqual(first.background_delivered, second.background_delivered)
        first_queues = [list(queue) for queue in first.node_queues]
        second_queues = [list(queue) for queue in second.node_queues]
        self.assertEqual(first_queues, second_queues)

    def test_shared_queue_and_calibrated_service_contract(self) -> None:
        self.assertEqual(len(self.env.node_queues), 45)
        self.assertEqual(self.config.simulation.max_queue_packets, 100)
        self.assertEqual(self.config.simulation.service_packets_per_slot, 36)
        # 300 Mbps is retained only for the 1500-byte packet transmission delay.
        transmission_ms = 1500 * 8 / (300 * 1_000_000) * 1000
        self.assertAlmostEqual(transmission_ms, 0.04)

    def test_queue_overflow_is_counted_separately(self) -> None:
        constrained = replace(
            self.config.simulation,
            max_queue_packets=1,
            service_packets_per_slot=1,
        )
        env = LEORoutingEnvironment(constrained, 180.0)
        env.reset(77)
        env.reset_measurement_counters()
        for _ in range(3):
            env.advance_background()
        self.assertGreater(env.background_queue_dropped, 0)
        self.assertEqual(
            env.background_dropped,
            env.background_queue_dropped
            + env.background_ttl_dropped
            + env.background_no_route_dropped,
        )

    def test_training_checkpoint_resume_matches_uninterrupted_run(self) -> None:
        two_episode_config = replace(
            self.config,
            learning=replace(
                self.config.learning,
                training_episodes=2,
                evaluation_episodes=0,
            ),
        )
        uninterrupted_model, uninterrupted_rows = train_variant(
            "dqn", two_episode_config
        )

        one_episode_config = replace(
            two_episode_config,
            learning=replace(two_episode_config.learning, training_episodes=1),
        )
        test_output_root = Path("outputs")
        test_output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_output_root) as directory:
            checkpoint_path = Path(directory) / "resume.pth"
            train_variant(
                "dqn",
                one_episode_config,
                checkpoint_path=checkpoint_path,
                checkpoint_every=1,
            )
            resume_state = torch.load(checkpoint_path, map_location="cpu")
            resumed_model, resumed_rows = train_variant(
                "dqn",
                two_episode_config,
                resume_state=resume_state,
            )

        self.assertEqual(resumed_rows, uninterrupted_rows)
        self.assertEqual(resumed_rows[0]["pre_focal_warmup_slots"], 1)
        self.assertEqual(resumed_rows[0]["background_policy_epsilon"], 0.0)
        self.assertGreater(resumed_rows[0]["pre_focal_average_queue_packets"], 0)
        for name, expected in uninterrupted_model.state_dict().items():
            self.assertTrue(torch.equal(resumed_model.state_dict()[name], expected), name)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_gat_lstm_cuda_checkpoint_can_resume(self) -> None:
        cuda = torch.device("cuda")
        two_episode_config = replace(
            self.config,
            learning=replace(
                self.config.learning,
                training_episodes=2,
                evaluation_episodes=0,
            ),
        )
        one_episode_config = replace(
            two_episode_config,
            learning=replace(two_episode_config.learning, training_episodes=1),
        )
        test_output_root = Path("outputs")
        test_output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_output_root) as directory:
            checkpoint_path = Path(directory) / "cuda_resume.pth"
            train_variant(
                "gat_lstm_dqn",
                one_episode_config,
                checkpoint_path=checkpoint_path,
                checkpoint_every=1,
                device=cuda,
            )
            resume_state = torch.load(checkpoint_path, map_location="cpu")
            resumed_model, resumed_rows = train_variant(
                "gat_lstm_dqn",
                two_episode_config,
                resume_state=resume_state,
                device=cuda,
            )

        self.assertEqual(len(resumed_rows), 2)
        self.assertEqual(next(resumed_model.parameters()).device.type, "cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_gat_lstm_cuda_chunking_preserves_outputs_and_actions(self) -> None:
        torch.manual_seed(11)
        device = torch.device("cuda")
        model = UnifiedRoutingQNetwork("gat_lstm_dqn", 6, 64, 128).to(device)
        model.train()
        states = torch.randn(
            CUDA_BACKGROUND_BATCH_SIZE + 17,
            self.config.observation_window,
            self.config.simulation.num_satellites,
            6,
        )
        adjacency = self.env.adjacency_tensor(0).to(device)

        with torch.no_grad():
            full = model(states.to(device), adjacency).cpu()
            chunked = torch.cat(
                [
                    model(
                        states[start : start + CUDA_BACKGROUND_BATCH_SIZE].to(device),
                        adjacency,
                    ).cpu()
                    for start in range(0, len(states), CUDA_BACKGROUND_BATCH_SIZE)
                ],
                dim=0,
            )

        self.assertTrue(torch.allclose(full, chunked, rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.equal(full.argmax(dim=1), chunked.argmax(dim=1)))

    def test_evaluation_logs_pre_focal_queue_and_each_completed_episode(self) -> None:
        model = UnifiedRoutingQNetwork("dqn", 6, 24, 24)
        streamed_rows = []
        rows = evaluate_variant(
            "dqn", model, self.config, progress_callback=streamed_rows.append
        )
        self.assertEqual(rows, streamed_rows)
        self.assertEqual(len(rows), self.config.learning.evaluation_episodes)
        self.assertEqual(rows[0]["pre_focal_warmup_slots"], 1)
        self.assertEqual(rows[0]["background_policy_epsilon"], 0.0)
        self.assertGreater(rows[0]["pre_focal_average_queue_packets"], 0)


if __name__ == "__main__":
    unittest.main()

