"""Frozen-checkpoint, cohort-aware four-scenario robustness evaluation.

This module intentionally leaves the training environment and old evaluation
outputs unchanged. Packet histories are written as compressed CSV per run.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .config import SimulationConfig
from .environment import LEORoutingEnvironment, RoutingRequest
from .baseline_packet import PacketDecision, PacketTransitionEvent, RoutingSnapshot
from .models import VARIANTS
from .robustness_pilot import (
    PilotSettings, SCENARIOS, load_model, make_trace, phase_for, recovery_time,
    write_csv,
)
from .run import ModelBackgroundRouter


PACKET_FIELDS = (
    "packet_id", "traffic_seed", "traffic_class", "birth_phase", "birth_slot",
    "end_slot", "source", "destination", "outcome", "drop_reason", "last_node",
    "hops", "shortest_hops_at_birth", "extra_hops", "inter_plane_crossings",
    "path", "queue_wait_ms", "slot_sojourn_ms", "end_to_end_delay_ms",
    "propagation_delay_ms", "transmission_delay_ms", "processing_delay_ms",
    "physical_component_sum_ms",
)


@dataclass
class PacketRecord:
    packet_id: int
    measured: bool
    traffic_class: str
    birth_phase: str
    birth_slot: int
    source: int
    destination: int
    shortest_hops: Optional[int]
    next_eligible_slot: int
    path: List[int] = field(default_factory=list)
    hops: int = 0
    crossings: int = 0
    queue_wait_slots: int = 0
    propagation_ms: float = 0.0
    transmission_ms: float = 0.0
    processing_ms: float = 0.0


def shortest_hops(source: int, destination: int, neighbors: List[List[int]]) -> Optional[int]:
    frontier = deque([(source, 0)])
    reached = {source}
    while frontier:
        node, depth = frontier.popleft()
        if node == destination:
            return depth
        for peer in neighbors[node]:
            if peer not in reached:
                reached.add(peer)
                frontier.append((peer, depth + 1))
    return None


class TrackedScenarioEnvironment(LEORoutingEnvironment):
    """Same FIFO/service/action sequence as the base environment, with event hooks."""

    def __init__(self, simulation, settings, trace, destinations, use_ftp, writer):
        super().__init__(simulation, settings.background_load_mbps)
        self.settings = settings
        self.trace = trace
        self.destinations = tuple(destinations)
        self.use_ftp = use_ftp
        self.writer = writer
        self.packet_id = 0
        self.records: Dict[int, PacketRecord] = {}
        self.active_sessions: List[List[int]] = []
        self.nhpp_injected = 0
        self.ftp_injected = 0
        self.events_by_slot: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.phase_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.class_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.delivered_sums: Dict[str, float] = defaultdict(float)
        self.shortest_cache: Dict[Tuple[int, int], Optional[int]] = {}

    def _new_packet(self, source: int, destination: int, kind: str) -> None:
        slot = self.step_index
        self.packet_id += 1
        measured = self.measurement_active
        if measured:
            self.background_generated += 1
            self.phase_counts[phase_for(slot, self.trace)]["generated"] += 1
            self.class_counts[kind]["generated"] += 1
        key = (slot, source)
        if measured and key not in self.shortest_cache:
            self.shortest_cache[key] = shortest_hops(
                source, destination, self.neighbors_at(slot)
            )
        self.records[self.packet_id] = PacketRecord(
            packet_id=self.packet_id, measured=measured, traffic_class=kind,
            birth_phase=phase_for(slot, self.trace), birth_slot=slot,
            source=source, destination=destination,
            shortest_hops=self.shortest_cache.get(key) if measured else None,
            next_eligible_slot=slot, path=[source],
        )
        packet = (destination, 0, 0, measured, self.packet_id)
        self._enqueue(source, packet)

    def _generate_background(self) -> None:
        slot = self.step_index
        if slot >= self.trace.horizon:
            raise RuntimeError("Formal scenario exceeded pre-generated traffic")
        for source, count in enumerate(self.trace.nhpp_counts[slot]):
            self.nhpp_injected += count
            for _ in range(count):
                self._new_packet(source, self.destinations[source], "nhpp")
        if not self.use_ftp:
            return
        for source in self.trace.ftp_starts.get(slot, ()):
            self.active_sessions.append([source, self.settings.ftp_file_packets])
        next_sessions = []
        for source, remaining in self.active_sessions:
            count = min(remaining, self.settings.ftp_packets_per_slot)
            self.ftp_injected += count
            for _ in range(count):
                self._new_packet(source, self.destinations[source], "ftp")
            if remaining > count:
                next_sessions.append([source, remaining - count])
        self.active_sessions = next_sessions

    def _write_record(self, record: PacketRecord, outcome: str, reason: str,
                      end_slot: Optional[int], last_node: int) -> None:
        if not record.measured:
            return
        delivered = outcome == "delivered"
        slot_sojourn = (
            (end_slot - record.birth_slot + 1) * self.config.slot_duration_s * 1000.0
            if delivered and end_slot is not None else None
        )
        shortest = record.shortest_hops
        row = {
            "packet_id": record.packet_id,
            "traffic_seed": self.settings.seed,
            "traffic_class": record.traffic_class,
            "birth_phase": record.birth_phase,
            "birth_slot": record.birth_slot,
            "end_slot": end_slot if end_slot is not None else "",
            "source": record.source,
            "destination": record.destination,
            "outcome": outcome,
            "drop_reason": reason,
            "last_node": last_node,
            "hops": record.hops,
            "shortest_hops_at_birth": shortest if shortest is not None else "",
            "extra_hops": record.hops - shortest if delivered and shortest is not None else "",
            "inter_plane_crossings": record.crossings,
            "path": "-".join(map(str, record.path)),
            "queue_wait_ms": round(record.queue_wait_slots * self.config.slot_duration_s * 1000.0, 6),
            "slot_sojourn_ms": round(slot_sojourn, 6) if slot_sojourn is not None else "",
            "end_to_end_delay_ms": round(slot_sojourn, 6) if slot_sojourn is not None else "",
            "propagation_delay_ms": round(record.propagation_ms, 6),
            "transmission_delay_ms": round(record.transmission_ms, 6),
            "processing_delay_ms": round(record.processing_ms, 6),
            "physical_component_sum_ms": round(
                record.propagation_ms + record.transmission_ms + record.processing_ms, 6
            ),
        }
        self.writer.writerow(row)
        self.phase_counts[record.birth_phase][outcome] += 1
        self.class_counts[record.traffic_class][outcome] += 1
        if end_slot is not None:
            event = self.events_by_slot[end_slot]
            event[outcome] += 1
            if outcome == "dropped":
                event[f"drop_{reason}"] += 1
            if delivered:
                event["delay_sum_ms"] += slot_sojourn
                event["hops_sum"] += record.hops
                event["crossings_sum"] += record.crossings
                self.delivered_sums["end_to_end_delay_ms"] += slot_sojourn
                self.delivered_sums["queue_wait_ms"] += (
                    record.queue_wait_slots * self.config.slot_duration_s * 1000.0
                )
                self.delivered_sums["propagation_delay_ms"] += record.propagation_ms
                self.delivered_sums["transmission_delay_ms"] += record.transmission_ms
                self.delivered_sums["processing_delay_ms"] += record.processing_ms
                self.delivered_sums["hops"] += record.hops
                self.delivered_sums["shortest_hops"] += shortest if shortest is not None else 0
                self.delivered_sums["crossings"] += record.crossings

    def _record_drop(self, packet, reason: str) -> None:
        super()._record_drop(packet, reason)
        record = self.records.pop(packet[4])
        self._write_record(record, "dropped", reason, self.step_index, record.path[-1])

    def _snapshot(self, neighbors):
        return RoutingSnapshot(
            slot=self.step_index,
            positions=tuple(self.positions(self.step_index)),
            queue_lengths=tuple(len(queue) for queue in self.node_queues),
            neighbors=tuple(tuple(row) for row in neighbors),
        )

    def _service_nodes(self, neighbors, background_router, forced_actions=None):
        if forced_actions:
            raise RuntimeError("Formal evaluation never routes focal packets")
        scheduled = []
        requests: List[RoutingRequest] = []
        packet_requests: List[PacketDecision] = []
        seen = set()
        for node, queue in enumerate(self.node_queues):
            for packet in list(queue)[:min(self.config.service_packets_per_slot, len(queue))]:
                scheduled.append((node, packet))
                request = (node, packet[0])
                if packet[2] < self.config.max_hops and neighbors[node]:
                    packet_requests.append(PacketDecision(
                        packet_id=packet[4], current=node, destination=packet[0],
                        hops=packet[2], slot=self.step_index,
                        feasible_neighbors=tuple(neighbors[node]),
                    ))
                if packet[2] < self.config.max_hops and request not in seen:
                    requests.append(request)
                    seen.add(request)
        per_packet = background_router is not None and hasattr(background_router, "route_packets")
        snapshot = None
        if per_packet:
            snapshot = self._snapshot(neighbors)
            policy_actions = (background_router.route_packets(packet_requests, snapshot)
                              if packet_requests else {})
            expected = {request.packet_id for request in packet_requests}
            if set(policy_actions) != expected:
                raise ValueError("Per-packet router must answer every eligible packet exactly once")
        else:
            policy_actions = (
                background_router(requests, self.step_index, neighbors)
                if background_router is not None and requests else {}
            )
        forwarded = []
        event_rows = []
        decisions_by_id = {decision.packet_id: decision for decision in packet_requests}
        positions = self.positions(self.step_index)
        transmission_ms = (
            self.config.packet_size_bytes * 8.0
            / (self.config.isl_capacity_mbps * 1_000_000.0) * 1000.0
        )
        for node, packet in scheduled:
            actual = self.node_queues[node].popleft()
            if actual != packet:
                raise RuntimeError("Shared FIFO queue order changed")
            destination, token, hops, measured, packet_id = packet
            if token != 0:
                raise RuntimeError("Unexpected focal packet")
            record = self.records[packet_id]
            record.queue_wait_slots += max(0, self.step_index - record.next_eligible_slot)
            if hops >= self.config.max_hops:
                self._record_drop(packet, "ttl")
                continue
            if per_packet and not neighbors[node]:
                self._record_drop(packet, "no_route")
                continue
            action = policy_actions.get(packet_id if per_packet else (node, destination))
            if per_packet and action not in neighbors[node]:
                raise ValueError(f"Per-packet router chose unavailable ISL {node}->{action}")
            if action not in neighbors[node]:
                action = self._shortest_next_hop(node, destination, neighbors)
            if action is None:
                self._record_drop(packet, "no_route")
                continue
            if per_packet:
                event_rows.append((decisions_by_id[packet_id], action,
                                   max(0, self.step_index - record.next_eligible_slot),
                                   action == destination))
            record.hops += 1
            record.crossings += int(node // 15 != action // 15)
            record.path.append(action)
            record.propagation_ms += (
                self.distance(positions[node], positions[action])
                / self.LIGHT_SPEED_KM_S * 1000.0
            )
            record.transmission_ms += transmission_ms
            record.processing_ms += self.config.processing_delay_ms
            if action == destination:
                if measured:
                    self.background_delivered += 1
                self.records.pop(packet_id)
                self._write_record(record, "delivered", "", self.step_index, action)
            else:
                record.next_eligible_slot = self.step_index + 1
                forwarded.append((action, (destination, 0, hops + 1, measured, packet_id)))
        dropped_forwarded = set()
        for target, packet in forwarded:
            if not self._enqueue(target, packet):
                dropped_forwarded.add(packet[4])
        if per_packet and hasattr(background_router, "on_slot_end"):
            events = [PacketTransitionEvent(
                decision=decision, action=action, queue_wait_slots=wait,
                delivered=delivered,
                dropped=(decision.packet_id in dropped_forwarded or
                         (decision.hops + 1 >= self.config.max_hops and not delivered)),
                next_node=action,
            ) for decision, action, wait, delivered in event_rows]
            background_router.on_slot_end(events, snapshot, self._snapshot(neighbors))
        return set(), set()

    def finish(self) -> int:
        remaining = list(self.records.values())
        for record in remaining:
            self._write_record(record, "in_flight_at_horizon", "", None, record.path[-1])
        self.records.clear()
        return sum(record.measured for record in remaining)


def checkpoint_path(seed: int, variant: str) -> Path:
    root = (
        Path("outputs/ablation_seed11_load240_tw5_2000") if seed == 11
        else Path(f"outputs/ablation_multiseed_load240_tw5_2000_eval200/seed{seed}")
    )
    return root / variant / f"{variant}_seed{seed}.pth"


def trace_digest(trace) -> str:
    raw = json.dumps({
        "nhpp_counts": trace.nhpp_counts,
        "ftp_starts": trace.ftp_starts,
        "same": trace.same_region_destinations,
        "cross": trace.cross_region_destinations,
        "horizon": trace.horizon,
    }, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def run_scenario(name, cross, use_ftp, variant, model, checkpoint_seed, settings, simulation,
                 trace, trace_hash, output, checkpoint_override=None):
    scenario_dir = output / f"checkpoint{checkpoint_seed}_traffic{settings.seed}" / variant / name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    summary_path = scenario_dir / "summary_FORMAL.json"
    if summary_path.exists():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_checkpoint = str(checkpoint_override or checkpoint_path(checkpoint_seed, variant))
        if (prior.get("status") == "FORMAL_COMPLETE" and
                prior.get("trace_sha256") == trace_hash and
                prior.get("checkpoint") == expected_checkpoint and
                prior.get("background_load_mbps") == settings.background_load_mbps and
                prior.get("warmup_slots") == settings.warmup_slots and
                prior.get("baseline_slots") == settings.baseline_slots and
                prior.get("burst_slots") == settings.burst_slots and
                prior.get("recovery_slots") == settings.recovery_slots):
            if (prior.get("recovery_definition", "legacy") != settings.recovery_definition or
                    prior.get("fixed_measurement_slots") != settings.measurement_slots):
                raise RuntimeError(f"Existing recovery protocol is incompatible: {scenario_dir}")
            if (scenario_dir / "packets_FORMAL.csv.gz").exists() and (
                scenario_dir / "slot_metrics_FORMAL.csv"
            ).exists():
                return prior
        raise RuntimeError(f"Existing output is incomplete or incompatible: {scenario_dir}")
    destinations = trace.cross_region_destinations if cross else trace.same_region_destinations
    packet_path = scenario_dir / "packets_FORMAL.csv.gz"
    with gzip.open(packet_path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PACKET_FIELDS)
        writer.writeheader()
        env = TrackedScenarioEnvironment(
            simulation, settings, trace, destinations, use_ftp, writer
        )
        env.reset(settings.seed * 100_000 + 9_005)
        if hasattr(model, "route_packets"):
            if hasattr(model, "reset_runtime"):
                model.reset_runtime()
            router = model
        else:
            router = ModelBackgroundRouter(
                model, env, observation_window=5, epsilon=0.0,
                rng=random.Random(settings.seed * 100_000 + 9_006),
            )
        for _ in range(settings.warmup_slots):
            env.advance_background(router)
        env.reset_measurement_counters()
        rows = []
        for _ in range(settings.warmup_slots, trace.horizon):
            slot = env.step_index
            if (slot - settings.warmup_slots) % 50 == 0:
                import datetime
                heartbeat = scenario_dir / "scenario_progress.json"
                pending_heartbeat = heartbeat.with_suffix(".tmp")
                pending_heartbeat.write_text(json.dumps({
                    "status": "running", "variant": variant, "scenario": name,
                    "measurement_slots_completed": slot-settings.warmup_slots,
                    "measurement_slots_total": trace.horizon-settings.warmup_slots,
                    "measurement_time_s": (slot-settings.warmup_slots)*simulation.slot_duration_s,
                    "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }), encoding="utf-8")
                pending_heartbeat.replace(heartbeat)
            before_generated = env.background_generated
            before_ftp = env.ftp_injected
            average_queue = env.advance_background(router)
            if not math.isfinite(average_queue) or not 0 <= average_queue <= simulation.max_queue_packets:
                raise RuntimeError(f"Invalid queue length at slot {slot}")
            if any(len(queue) > simulation.max_queue_packets for queue in env.node_queues):
                raise RuntimeError(f"Node queue overflow at slot {slot}")
            event = env.events_by_slot.pop(slot, {})
            delivered = int(event.get("delivered", 0))
            rows.append({
                "slot": slot,
                "phase": phase_for(slot, trace),
                "average_queue_packets": round(average_queue, 6),
                "generated_packets": env.background_generated - before_generated,
                "ftp_injected_packets": env.ftp_injected - before_ftp,
                "delivered_packets": delivered,
                "dropped_packets": int(event.get("dropped", 0)),
                "queue_dropped_packets": int(event.get("drop_queue", 0)),
                "ttl_dropped_packets": int(event.get("drop_ttl", 0)),
                "no_route_dropped_packets": int(event.get("drop_no_route", 0)),
                "throughput_mbps": round(
                    delivered * simulation.packet_size_bytes * 8.0
                    / simulation.slot_duration_s / 1_000_000.0, 6
                ),
                "mean_delivered_delay_ms": round(event.get("delay_sum_ms", 0.0) / delivered, 6)
                if delivered else "",
                "mean_delivered_hops": round(event.get("hops_sum", 0.0) / delivered, 6)
                if delivered else "",
                "mean_delivered_inter_plane_crossings": round(
                    event.get("crossings_sum", 0.0) / delivered, 6
                ) if delivered else "",
            })
        remaining = env.finish()
    expected_nhpp = sum(
        sum(trace.nhpp_counts[slot]) for slot in range(settings.warmup_slots, trace.horizon)
    )
    expected_ftp = (
        sum(map(len, trace.ftp_starts.values())) * settings.ftp_file_packets
        if use_ftp else 0
    )
    if env.background_generated != expected_nhpp + expected_ftp:
        raise RuntimeError("Generated packets differ from paired trace")
    if env.ftp_injected != expected_ftp or env.active_sessions:
        raise RuntimeError("FTP session injection incomplete")
    if env.background_delivered + env.background_dropped + remaining != env.background_generated:
        raise RuntimeError("Packet conservation failed")
    if (env.background_queue_dropped + env.background_ttl_dropped
            + env.background_no_route_dropped != env.background_dropped):
        raise RuntimeError("Drop-reason conservation failed")
    if sum(row["generated_packets"] for row in rows) != env.background_generated:
        raise RuntimeError("Slot generated counts do not sum to scenario total")
    if sum(row["delivered_packets"] for row in rows) != env.background_delivered:
        raise RuntimeError("Slot delivered counts do not sum to scenario total")
    if sum(row["dropped_packets"] for row in rows) != env.background_dropped:
        raise RuntimeError("Slot dropped counts do not sum to scenario total")
    recovery, q0, threshold, recovery_status, burst_peak = recovery_time(
        rows, trace, settings, simulation.slot_duration_s
    )
    if settings.recovery_definition == "teacher_20261023":
        from .revision_burst_analysis import recovery as teacher_recovery, rolling
        queues = [float(row["average_queue_packets"]) for row in rows]
        record = teacher_recovery(queues, settings.baseline_slots,
                                  settings.baseline_slots + settings.burst_slots,
                                  dt=simulation.slot_duration_s)
        recovery, q0, threshold, recovery_status = (
            record["recovery_time_s"], record["q_pre"], record["threshold"], record["status"])
        burst_peak = max(v for v in rolling(queues, 10)[settings.baseline_slots:]
                         if math.isfinite(v))
    elif settings.recovery_definition != "legacy":
        raise ValueError(f"Unknown recovery definition: {settings.recovery_definition}")
    if not use_ftp:
        recovery = None
        recovery_status = "not_applicable"
    measured_slots = len(rows)
    delivered = env.background_delivered
    duration = measured_slots * simulation.slot_duration_s
    summary = {
        "status": "FORMAL_COMPLETE",
        "variant": variant, "scenario": name, "traffic_seed": settings.seed,
        "checkpoint_seed": checkpoint_seed,
        "checkpoint": str(checkpoint_override or checkpoint_path(checkpoint_seed, variant)),
        "checkpoint_training_load_mbps": 240.0,
        "background_load_mbps": settings.background_load_mbps,
        "trace_sha256": trace_hash,
        "warmup_slots": settings.warmup_slots,
        "baseline_slots": settings.baseline_slots,
        "burst_slots": settings.burst_slots,
        "ftp_tail_slots": trace.recovery_start - trace.burst_end,
        "recovery_slots": settings.recovery_slots,
        "measurement_slots": measured_slots,
        "fixed_measurement_slots": settings.measurement_slots,
        "recovery_definition": settings.recovery_definition,
        "ftp_sessions": sum(map(len, trace.ftp_starts.values())) if use_ftp else 0,
        "nhpp_injected_packets": expected_nhpp,
        "ftp_injected_packets": expected_ftp,
        "generated_packets": env.background_generated,
        "delivered_packets": delivered,
        "dropped_packets": env.background_dropped,
        "queue_dropped_packets": env.background_queue_dropped,
        "ttl_dropped_packets": env.background_ttl_dropped,
        "no_route_dropped_packets": env.background_no_route_dropped,
        "in_flight_at_horizon": remaining,
        "throughput_mbps": delivered * simulation.packet_size_bytes * 8.0 / duration / 1_000_000.0,
        "packet_loss_rate": env.background_dropped / env.background_generated,
        "in_flight_fraction": remaining / env.background_generated,
        "end_to_end_delay_ms": env.delivered_sums["end_to_end_delay_ms"] / delivered if delivered else None,
        "queue_wait_ms": env.delivered_sums["queue_wait_ms"] / delivered if delivered else None,
        "propagation_delay_ms": env.delivered_sums["propagation_delay_ms"] / delivered if delivered else None,
        "transmission_delay_ms": env.delivered_sums["transmission_delay_ms"] / delivered if delivered else None,
        "processing_delay_ms": env.delivered_sums["processing_delay_ms"] / delivered if delivered else None,
        "average_queue_packets": statistics.mean(float(row["average_queue_packets"]) for row in rows),
        "mean_hops": env.delivered_sums["hops"] / delivered if delivered else None,
        "mean_shortest_hops_at_birth": env.delivered_sums["shortest_hops"] / delivered if delivered else None,
        "mean_inter_plane_crossings": env.delivered_sums["crossings"] / delivered if delivered else None,
        "recovery_time_s": recovery,
        "recovery_status": recovery_status,
        "recovery_baseline_q0": q0,
        "recovery_threshold": threshold,
        "recovery_burst_peak_rolling_queue": burst_peak,
        "birth_phase_counts": dict(env.phase_counts),
        "traffic_class_counts": dict(env.class_counts),
        "delay_definition": "Mean of delivered measured packets: (delivery_slot-birth_slot+1)*20ms; physical link components are diagnostic, not added again.",
        "loss_definition": "Dropped measured packets / generated measured packets; in-flight packets are separately censored, not counted as loss.",
    }
    write_csv(scenario_dir / "slot_metrics_FORMAL.csv", rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (scenario_dir / "scenario_progress.json").write_text(json.dumps({
        "status": "complete", "measurement_slots_completed": measured_slots,
        "measurement_slots_total": measured_slots,
        "measurement_time_s": duration,
    }), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-seeds", "--seeds", dest="checkpoint_seeds",
                        type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--traffic-seeds", type=int, nargs="+",
                        default=[101, 202, 303, 404, 505])
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS),
                        default=["dqn", "lstm_dqn", "gat_dqn", "gat_lstm_dqn"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup-slots", type=int, default=30)
    parser.add_argument("--baseline-slots", type=int, default=250)
    parser.add_argument("--burst-slots", type=int, default=250)
    parser.add_argument("--recovery-slots", type=int, default=500)
    args = parser.parse_args()
    if len(set(args.checkpoint_seeds)) != len(args.checkpoint_seeds):
        raise ValueError("Duplicate checkpoint seeds")
    if len(set(args.traffic_seeds)) != len(args.traffic_seeds):
        raise ValueError("Duplicate traffic seeds")
    if len(args.checkpoint_seeds) != len(args.traffic_seeds):
        raise ValueError("Checkpoint and traffic seed lists must have equal length")
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("Duplicate model variants")
    if min(args.warmup_slots, args.baseline_slots, args.burst_slots, args.recovery_slots) <= 0:
        raise ValueError("All durations must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "FORMAL_COMPLETE.ok").unlink(missing_ok=True)
    (output / "FORMAL_FAILED.txt").unlink(missing_ok=True)
    simulation = SimulationConfig()
    all_summaries = []
    try:
        for checkpoint_seed, seed in zip(args.checkpoint_seeds, args.traffic_seeds):
            settings = PilotSettings(
                seed=seed, warmup_slots=args.warmup_slots,
                baseline_slots=args.baseline_slots, burst_slots=args.burst_slots,
                recovery_slots=args.recovery_slots,
            )
            trace = make_trace(settings, simulation)
            trace_hash = trace_digest(trace)
            seed_dir = output / f"checkpoint{checkpoint_seed}_traffic{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "trace_manifest.json").write_text(json.dumps({
                "settings": asdict(settings),
                "simulation": {
                    "packet_size_bytes": simulation.packet_size_bytes,
                    "service_packets_per_slot": simulation.service_packets_per_slot,
                    "max_queue_packets": simulation.max_queue_packets,
                    "slot_duration_s": simulation.slot_duration_s,
                    "isl_capacity_mbps": simulation.isl_capacity_mbps,
                },
                "trace_sha256": trace_hash,
                "burst_start": trace.burst_start, "burst_end": trace.burst_end,
                "recovery_start": trace.recovery_start, "horizon": trace.horizon,
                "ftp_starts": {str(k): list(v) for k, v in trace.ftp_starts.items()},
                "same_region_destinations": list(trace.same_region_destinations),
                "cross_region_destinations": list(trace.cross_region_destinations),
                "nhpp_counts": trace.nhpp_counts,
            }, ensure_ascii=False), encoding="utf-8")
            for variant in args.variants:
                path = checkpoint_path(checkpoint_seed, variant)
                if not path.is_file():
                    raise FileNotFoundError(path)
                model = load_model(path, variant, torch.device(args.device))
                for name, cross, use_ftp in SCENARIOS:
                    print(f"[{checkpoint_seed}/{seed}/{variant}/{name}] starting", flush=True)
                    result = run_scenario(
                        name, cross, use_ftp, variant, model, checkpoint_seed, settings,
                        simulation, trace, trace_hash, output,
                    )
                    all_summaries.append(result)
                    print(
                        f"[{checkpoint_seed}/{seed}/{variant}/{name}] complete: "
                        f"delivered={result['delivered_packets']} "
                        f"dropped={result['dropped_packets']}", flush=True,
                    )
                    (output / "progress.json").write_text(json.dumps({
                        "completed": len(all_summaries),
                        "total": len(args.checkpoint_seeds) * len(args.variants) * len(SCENARIOS),
                        "last_checkpoint_seed": checkpoint_seed,
                        "last_traffic_seed": seed, "last_variant": variant,
                        "last_scenario": name,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
        expected_count = len(args.checkpoint_seeds) * len(args.variants) * len(SCENARIOS)
        if len(all_summaries) != expected_count:
            raise RuntimeError("Formal run count incomplete")
        for seed in args.traffic_seeds:
            by_seed = [row for row in all_summaries if row["traffic_seed"] == seed]
            for name, _, _ in SCENARIOS:
                if len({row["generated_packets"] for row in by_seed if row["scenario"] == name}) != 1:
                    raise RuntimeError("Model variants did not receive matched traffic")
        write_csv(output / "summary_FORMAL.csv", [
            {key: value for key, value in result.items()
             if key not in ("birth_phase_counts", "traffic_class_counts")}
            for result in all_summaries
        ])
        (output / "FORMAL_COMPLETE.ok").write_text(
            f"All {expected_count} paired frozen-checkpoint scenarios completed.\n",
            encoding="utf-8",
        )
    except Exception as error:
        (output / "FORMAL_FAILED.txt").write_text(repr(error) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
