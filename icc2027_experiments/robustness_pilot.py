"""PILOT_ONLY paired cross-region and FTP-burst pipeline check.

This module loads frozen checkpoints; it never trains or changes the standard
training/evaluation code path. Its short windows are not paper statistics.
"""

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from .config import SimulationConfig
from .environment import LEORoutingEnvironment
from .models import VARIANTS, UnifiedRoutingQNetwork
from .run import ModelBackgroundRouter


SCENARIOS = (
    ("S1_non_cross_normal", False, False),
    ("S2_cross_normal", True, False),
    ("S3_non_cross_ftp", False, True),
    ("S4_cross_ftp", True, True),
)


@dataclass(frozen=True)
class PilotSettings:
    seed: int = 11
    background_load_mbps: float = 120.0
    ftp_sessions_per_second: float = 3.75
    ftp_file_packets: int = 1334
    ftp_packets_per_slot: int = 20
    warmup_slots: int = 30
    baseline_slots: int = 60
    burst_slots: int = 80
    recovery_slots: int = 120
    recovery_rolling_slots: int = 20
    recovery_stable_slots: int = 50
    measurement_slots: int | None = None
    recovery_definition: str = "legacy"


@dataclass(frozen=True)
class PairedTrace:
    nhpp_counts: Tuple[Tuple[int, ...], ...]
    ftp_starts: Dict[int, Tuple[int, ...]]
    same_region_destinations: Tuple[int, ...]
    cross_region_destinations: Tuple[int, ...]
    burst_start: int
    burst_end: int
    recovery_start: int
    horizon: int


def poisson(rng: random.Random, rate: float) -> int:
    threshold = math.exp(-rate)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def paired_destinations(seed: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    rng = random.Random(seed * 100_000 + 9_001)
    same: List[int] = []
    cross: List[int] = []
    for source in range(45):
        plane, satellite = divmod(source, 15)
        within_index = int(rng.random() * 14)
        within_satellite = within_index + (within_index >= satellite)
        same.append(plane * 15 + within_satellite)
        other_plane = (plane + 1 + (source + seed) % 2) % 3
        other_satellite = int(rng.random() * 15)
        cross.append(other_plane * 15 + other_satellite)
    assert all(source != destination for source, destination in enumerate(same))
    assert all(source // 15 == destination // 15 for source, destination in enumerate(same))
    assert all(source // 15 != destination // 15 for source, destination in enumerate(cross))
    return tuple(same), tuple(cross)


def make_trace(settings: PilotSettings, simulation: SimulationConfig) -> PairedTrace:
    burst_start = settings.warmup_slots + settings.baseline_slots
    burst_end = burst_start + settings.burst_slots
    file_slots = math.ceil(settings.ftp_file_packets / settings.ftp_packets_per_slot)
    arrival_rng = random.Random(settings.seed * 100_000 + 9_002)
    source_rng = random.Random(settings.seed * 100_000 + 9_003)
    starts: Dict[int, Tuple[int, ...]] = {}
    busy_until = [0] * simulation.num_satellites
    for slot in range(burst_start, burst_end):
        events = poisson(
            arrival_rng,
            settings.ftp_sessions_per_second * simulation.slot_duration_s,
        )
        sources = []
        for _ in range(events):
            idle = [source for source, end in enumerate(busy_until) if end <= slot]
            if not idle:
                raise RuntimeError("No idle source is available for an FTP session")
            source = idle[source_rng.randrange(len(idle))]
            busy_until[source] = slot + file_slots
            sources.append(source)
        if sources:
            starts[slot] = tuple(sources)

    last_injection_exclusive = max(
        (slot + file_slots for slot in starts), default=burst_end
    )
    recovery_start = max(burst_end, last_injection_exclusive)
    horizon = recovery_start + settings.recovery_slots
    if settings.measurement_slots is not None:
        horizon = settings.warmup_slots + settings.measurement_slots
        if recovery_start >= horizon:
            raise ValueError("Fixed observation horizon must include all FTP injection and post-burst follow-up")

    nhpp_rng = random.Random(settings.seed * 100_000 + 9_004)
    base_rate = (
        settings.background_load_mbps * 1_000_000.0
        * simulation.slot_duration_s
        / (simulation.packet_size_bytes * 8.0)
        / simulation.num_satellites
    )
    counts = []
    for slot in range(horizon):
        slot_counts = []
        for source in range(simulation.num_satellites):
            phase = 2.0 * math.pi * source / simulation.num_satellites
            modulation = 1.0 + 0.35 * math.sin(
                2.0 * math.pi * slot / simulation.nhpp_period_steps + phase
            )
            slot_counts.append(poisson(nhpp_rng, max(0.0, base_rate * modulation)))
        counts.append(tuple(slot_counts))
    same, cross = paired_destinations(settings.seed)
    return PairedTrace(
        nhpp_counts=tuple(counts), ftp_starts=starts,
        same_region_destinations=same, cross_region_destinations=cross,
        burst_start=burst_start, burst_end=burst_end,
        recovery_start=recovery_start, horizon=horizon,
    )


class ScenarioEnvironment(LEORoutingEnvironment):
    def __init__(
        self, simulation: SimulationConfig, settings: PilotSettings,
        trace: PairedTrace, destinations: Sequence[int], use_ftp: bool,
    ) -> None:
        super().__init__(simulation, settings.background_load_mbps)
        self.settings = settings
        self.trace = trace
        self.destinations = tuple(destinations)
        self.use_ftp = use_ftp
        self.active_sessions: List[List[int]] = []
        self.nhpp_injected = 0
        self.ftp_injected = 0

    def _generate_background(self) -> None:
        slot = self.step_index
        if slot >= self.trace.horizon:
            raise RuntimeError("Scenario exceeded the pre-generated traffic trace")
        for source, count in enumerate(self.trace.nhpp_counts[slot]):
            destination = self.destinations[source]
            self.nhpp_injected += count
            for _ in range(count):
                packet = (destination, 0, 0, self.measurement_active)
                if self.measurement_active:
                    self.background_generated += 1
                self._enqueue(source, packet)

        if not self.use_ftp:
            return
        for source in self.trace.ftp_starts.get(slot, ()):
            self.active_sessions.append([source, self.settings.ftp_file_packets])
        next_sessions: List[List[int]] = []
        for source, remaining in self.active_sessions:
            count = min(remaining, self.settings.ftp_packets_per_slot)
            destination = self.destinations[source]
            self.ftp_injected += count
            for _ in range(count):
                packet = (destination, 0, 0, self.measurement_active)
                if self.measurement_active:
                    self.background_generated += 1
                self._enqueue(source, packet)
            if remaining > count:
                next_sessions.append([source, remaining - count])
        self.active_sessions = next_sessions


def load_model(path: Path, variant: str, device: torch.device) -> UnifiedRoutingQNetwork:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = payload["config"]
    if payload["variant"] != variant or config["offered_load_mbps"] != 240.0:
        raise ValueError(f"Not the frozen 240 Mbps {variant} checkpoint: {path}")
    if config["observation_window"] != 5:
        raise ValueError("Pilot requires an observation window of five")
    if config["simulation"]["service_packets_per_slot"] != 36:
        raise ValueError("Checkpoint has an unexpected service rate")
    learning = config["learning"]
    model = UnifiedRoutingQNetwork(
        variant, feature_dim=6, hidden_dim=learning["hidden_dim"],
        lstm_hidden_dim=learning["lstm_hidden_dim"],
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model


def phase_for(slot: int, trace: PairedTrace) -> str:
    if slot < trace.burst_start:
        return "baseline"
    if slot < trace.burst_end:
        return "burst"
    if slot < trace.recovery_start:
        return "ftp_tail"
    return "recovery"


def recovery_time(
    rows: Sequence[Dict[str, object]], trace: PairedTrace,
    settings: PilotSettings, slot_duration_s: float,
) -> Tuple[float | None, float, float, str, float]:
    baseline = [
        float(row["average_queue_packets"])
        for row in rows if row["phase"] == "baseline"
    ]
    if not baseline:
        raise RuntimeError("Missing baseline queue samples")
    q0 = statistics.mean(baseline[-min(100, len(baseline)):])
    threshold = q0 + max(0.10 * q0, 2.0)
    queues = [float(row["average_queue_packets"]) for row in rows]
    moving_averages = [
        statistics.mean(queues[max(0, index - settings.recovery_rolling_slots + 1):index + 1])
        for index in range(len(rows))
    ]
    burst_peak = max(
        moving_averages[index]
        for index, row in enumerate(rows)
        if row["phase"] in ("burst", "ftp_tail")
    )
    if burst_peak <= threshold:
        return None, q0, threshold, "no_queue_excursion", burst_peak
    stable = 0
    for index, row in enumerate(rows):
        if int(row["slot"]) < trace.recovery_start:
            continue
        stable = stable + 1 if moving_averages[index] <= threshold else 0
        if stable >= settings.recovery_stable_slots:
            elapsed_slots = int(row["slot"]) - trace.recovery_start + 1
            return elapsed_slots * slot_duration_s, q0, threshold, "recovered", burst_peak
    return None, q0, threshold, "right_censored", burst_peak


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_scenario(
    name: str, cross: bool, use_ftp: bool,
    variant: str, model: UnifiedRoutingQNetwork,
    settings: PilotSettings, simulation: SimulationConfig,
    trace: PairedTrace, output: Path,
) -> Dict[str, object]:
    destinations = (
        trace.cross_region_destinations if cross else trace.same_region_destinations
    )
    env = ScenarioEnvironment(simulation, settings, trace, destinations, use_ftp)
    env.reset(settings.seed * 100_000 + 9_005)
    router = ModelBackgroundRouter(
        model, env, observation_window=5, epsilon=0.0,
        rng=random.Random(settings.seed * 100_000 + 9_006),
    )
    for _ in range(settings.warmup_slots):
        env.advance_background(router)
    env.reset_measurement_counters()
    rows: List[Dict[str, object]] = []
    for _ in range(settings.warmup_slots, trace.horizon):
        slot = env.step_index
        before = (
            env.background_generated, env.background_delivered,
            env.background_dropped, env.ftp_injected,
        )
        average_queue = env.advance_background(router)
        if not math.isfinite(average_queue) or not 0 <= average_queue <= simulation.max_queue_packets:
            raise RuntimeError(f"Invalid shared queue length at slot {slot}")
        if any(len(queue) > simulation.max_queue_packets for queue in env.node_queues):
            raise RuntimeError(f"Node queue exceeded capacity at slot {slot}")
        rows.append({
            "slot": slot,
            "phase": phase_for(slot, trace),
            "average_queue_packets": round(average_queue, 6),
            "generated_packets": env.background_generated - before[0],
            "delivered_packets": env.background_delivered - before[1],
            "dropped_packets": env.background_dropped - before[2],
            "ftp_injected_packets": env.ftp_injected - before[3],
        })
    expected_nhpp = sum(
        sum(trace.nhpp_counts[slot])
        for slot in range(settings.warmup_slots, trace.horizon)
    )
    expected_ftp = (
        sum(map(len, trace.ftp_starts.values())) * settings.ftp_file_packets
        if use_ftp else 0
    )
    if env.background_generated != expected_nhpp + expected_ftp:
        raise RuntimeError("Generated-packet count does not match the paired trace")
    if env.ftp_injected != expected_ftp or env.active_sessions:
        raise RuntimeError("FTP sessions did not finish injecting")
    if env.background_delivered + env.background_dropped > env.background_generated:
        raise RuntimeError("Delivered plus dropped exceeds generated")
    if (
        env.background_queue_dropped + env.background_ttl_dropped
        + env.background_no_route_dropped != env.background_dropped
    ):
        raise RuntimeError("Drop-reason counters do not sum to total drops")

    observed_recovery, q0, threshold, recovery_status, burst_peak = recovery_time(
        rows, trace, settings, simulation.slot_duration_s
    )
    if not use_ftp:
        observed_recovery = None
        recovery_status = "not_applicable"
    scenario_dir = output / variant / name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    write_csv(scenario_dir / "slot_metrics_PILOT_ONLY.csv", rows)
    result: Dict[str, object] = {
        "variant": variant,
        "scenario": name,
        "status": "PILOT_ONLY",
        "seed": settings.seed,
        "checkpoint_training_load_mbps": 240.0,
        "background_load_mbps": settings.background_load_mbps,
        "flow_count": simulation.num_satellites,
        "nhpp_injected_packets_measured": expected_nhpp,
        "ftp_sessions": sum(map(len, trace.ftp_starts.values())) if use_ftp else 0,
        "ftp_injected_packets": expected_ftp,
        "generated_packets": env.background_generated,
        "delivered_packets": env.background_delivered,
        "dropped_packets": env.background_dropped,
        "queue_dropped_packets": env.background_queue_dropped,
        "ttl_dropped_packets": env.background_ttl_dropped,
        "no_route_dropped_packets": env.background_no_route_dropped,
        "average_queue_packets": round(statistics.mean(
            float(row["average_queue_packets"]) for row in rows
        ), 6),
        "peak_average_queue_packets": max(
            float(row["average_queue_packets"]) for row in rows
        ),
        "baseline_queue_q0": round(q0, 6),
        "recovery_queue_threshold": round(threshold, 6),
        "peak_rolling_queue_during_burst": round(burst_peak, 6),
        "recovery_time_s": observed_recovery,
        "recovery_status": recovery_status,
        "recovery_observation_censored": recovery_status == "right_censored",
        "measurement_slots": len(rows),
    }
    (scenario_dir / "summary_PILOT_ONLY.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS),
                        default=["gat_lstm_dqn"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup-slots", type=int, default=30)
    parser.add_argument("--baseline-slots", type=int, default=60)
    parser.add_argument("--burst-slots", type=int, default=80)
    parser.add_argument("--recovery-slots", type=int, default=120)
    args = parser.parse_args()
    settings = PilotSettings(
        seed=args.seed, warmup_slots=args.warmup_slots,
        baseline_slots=args.baseline_slots, burst_slots=args.burst_slots,
        recovery_slots=args.recovery_slots,
    )
    if min(settings.warmup_slots, settings.baseline_slots, settings.burst_slots,
           settings.recovery_slots) <= 0:
        raise ValueError("Pilot durations must be positive")
    simulation = SimulationConfig()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    trace = make_trace(settings, simulation)
    if not trace.ftp_starts:
        raise RuntimeError("Pilot trace has no FTP sessions; choose a longer burst")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    passed_marker = output / "PILOT_ONLY_passed.ok"
    passed_marker.unlink(missing_ok=True)
    (output / "pilot_config.json").write_text(
        json.dumps({
            "status": "PILOT_ONLY",
            "settings": asdict(settings),
            "simulation": {
                "packet_size_bytes": simulation.packet_size_bytes,
                "max_queue_packets": simulation.max_queue_packets,
                "service_packets_per_slot": simulation.service_packets_per_slot,
                "slot_duration_s": simulation.slot_duration_s,
            },
            "burst_start": trace.burst_start,
            "burst_end": trace.burst_end,
            "recovery_start": trace.recovery_start,
            "horizon": trace.horizon,
            "ftp_starts": {str(k): list(v) for k, v in trace.ftp_starts.items()},
            "same_region_destinations": list(trace.same_region_destinations),
            "cross_region_destinations": list(trace.cross_region_destinations),
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    results: List[Dict[str, object]] = []
    for variant in args.variants:
        checkpoint = (
            Path("outputs") / "ablation_seed11_load240_tw5_2000" / variant
            / f"{variant}_seed{settings.seed}.pth"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = load_model(checkpoint, variant, device)
        for name, cross, use_ftp in SCENARIOS:
            print(f"[{variant}/{name}] pilot starting", flush=True)
            result = run_scenario(
                name, cross, use_ftp, variant, model, settings,
                simulation, trace, output,
            )
            results.append(result)
            print(
                f"[{variant}/{name}] generated={result['generated_packets']} "
                f"dropped={result['dropped_packets']} "
                f"queue={result['average_queue_packets']}",
                flush=True,
            )
    by_name = defaultdict(list)
    for result in results:
        by_name[result["scenario"]].append(result)
    for result_group in by_name.values():
        if len({row["generated_packets"] for row in result_group}) != 1:
            raise RuntimeError("Model variants did not receive paired packet counts")
    for variant in args.variants:
        own = {row["scenario"]: row for row in results if row["variant"] == variant}
        if own["S1_non_cross_normal"]["generated_packets"] != own["S2_cross_normal"]["generated_packets"]:
            raise RuntimeError("Normal region scenarios have unequal injection")
        if own["S3_non_cross_ftp"]["generated_packets"] != own["S4_cross_ftp"]["generated_packets"]:
            raise RuntimeError("FTP region scenarios have unequal injection")
    write_csv(output / "summary_PILOT_ONLY.csv", results)
    passed_marker.write_text(
        "Paired four-scenario pipeline and counter invariants passed. "
        "Not a formal scientific result.\n", encoding="utf-8"
    )
    print(f"PILOT_ONLY checks passed: {output}", flush=True)


if __name__ == "__main__":
    main()
