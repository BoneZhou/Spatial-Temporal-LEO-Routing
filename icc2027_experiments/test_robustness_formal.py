import csv
import io
import math
import unittest
from dataclasses import replace

from .config import SimulationConfig
from .robustness_formal import PACKET_FIELDS, TrackedScenarioEnvironment
from .robustness_pilot import PilotSettings, make_trace


class TrackedRobustnessTests(unittest.TestCase):
    def test_formal_router_accepts_independent_same_destination_packets(self):
        config = SimulationConfig()
        settings = PilotSettings(
            seed=101, background_load_mbps=1.0,
            warmup_slots=1, baseline_slots=3, burst_slots=3,
            recovery_slots=6,
        )
        trace = make_trace(settings, config)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=PACKET_FIELDS)
        env = TrackedScenarioEnvironment(
            config, settings, trace, trace.same_region_destinations, False, writer
        )
        env.measurement_active = True
        env._new_packet(0, 10, "nhpp")
        env._new_packet(0, 10, "nhpp")

        class DistinctRouter:
            def __init__(self):
                self.ids = []
                self.events = []
                self.before = None
                self.after = None

            def route_packets(self, decisions, snapshot):
                self.ids = [decision.packet_id for decision in decisions]
                peers = snapshot.neighbors[0]
                return {self.ids[0]: peers[0], self.ids[1]: peers[1]}

            def on_slot_end(self, events, before, after):
                self.events = events
                self.before = before
                self.after = after

        router = DistinctRouter()
        env._service_nodes(env.neighbors_at(0), router)
        self.assertEqual(len(router.ids), 2)
        self.assertNotEqual(env.records[router.ids[0]].path[-1],
                            env.records[router.ids[1]].path[-1])
        self.assertEqual([event.decision.packet_id for event in router.events], router.ids)
        self.assertEqual(router.before.queue_lengths[0], 2)
        self.assertEqual(router.after.queue_lengths[0], 0)

    def test_packet_paths_and_conservation(self):
        config = replace(SimulationConfig(), service_packets_per_slot=36)
        settings = PilotSettings(
            seed=101, background_load_mbps=1.0,
            warmup_slots=1, baseline_slots=3, burst_slots=3,
            recovery_slots=6,
        )
        trace = make_trace(settings, config)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=PACKET_FIELDS)
        writer.writeheader()
        env = TrackedScenarioEnvironment(
            config, settings, trace, trace.same_region_destinations, False, writer
        )
        env.reset(1234)

        def shortest_router(requests, _step, neighbors):
            return {request: env._shortest_next_hop(*request, neighbors)
                    for request in requests}

        env.advance_background(shortest_router)
        env.reset_measurement_counters()
        for _ in range(1, trace.horizon):
            env.advance_background(shortest_router)
        remaining = env.finish()
        rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertEqual(len(rows), env.background_generated)
        self.assertEqual(
            env.background_generated,
            env.background_delivered + env.background_dropped + remaining,
        )
        for row in rows:
            path = [int(node) for node in row["path"].split("-")]
            self.assertEqual(len(path), int(row["hops"]) + 1)
            self.assertEqual(path[0], int(row["source"]))
            self.assertEqual(
                sum(a // 15 != b // 15 for a, b in zip(path, path[1:])),
                int(row["inter_plane_crossings"]),
            )
            if row["outcome"] == "delivered":
                self.assertEqual(path[-1], int(row["destination"]))
                self.assertTrue(math.isclose(
                    float(row["end_to_end_delay_ms"]),
                    (int(row["end_slot"]) - int(row["birth_slot"]) + 1) * 20.0,
                ))


if __name__ == "__main__":
    unittest.main()
