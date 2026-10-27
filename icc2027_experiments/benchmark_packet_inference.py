"""Benchmark frozen Proposed per-packet routing without changing evaluation data.

The timed region is the complete ``route_packets`` call: state/history creation,
batched network inference, CPU/GPU transfers, masking, and action selection.
It does not simulate a satellite processor or modify packet delay accounting.
"""

import argparse
import datetime
import gc
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch

from .baseline_packet import PacketDecision, RoutingSnapshot
from .config import SimulationConfig
from .environment import LEORoutingEnvironment
from .proposed_packet import ProposedPacketPolicy


def make_snapshot(config: SimulationConfig) -> RoutingSnapshot:
    environment = LEORoutingEnvironment(config, 240.0)
    neighbors = environment.neighbors_at(0)
    return RoutingSnapshot(
        slot=0,
        positions=tuple(environment.positions(0)),
        queue_lengths=tuple(50 for _ in range(config.num_satellites)),
        neighbors=tuple(tuple(row) for row in neighbors),
    )


def make_decisions(count: int, snapshot: RoutingSnapshot) -> list[PacketDecision]:
    decisions = []
    nodes = len(snapshot.neighbors)
    for index in range(count):
        current = index % nodes
        destination = (index * 7 + 13) % nodes
        if destination == current:
            destination = (destination + 1) % nodes
        feasible = snapshot.neighbors[current]
        if not feasible:
            raise RuntimeError(f"No feasible neighbor for satellite {current}")
        decisions.append(PacketDecision(
            packet_id=index + 1,
            current=current,
            destination=destination,
            hops=0,
            slot=0,
            feasible_neighbors=feasible,
        ))
    return decisions


def benchmark(policy: ProposedPacketPolicy, snapshot: RoutingSnapshot,
              decisions: list[PacketDecision], warmup: int,
              repeats: int) -> dict:
    count = len(decisions)

    def one_call() -> int:
        policy.reset_runtime()
        if policy.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter_ns()
        actions = policy.route_packets(decisions, snapshot)
        if policy.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter_ns() - start
        if len(actions) != count or any(
            actions[row.packet_id] not in row.feasible_neighbors
            for row in decisions
        ):
            raise RuntimeError("Incomplete or infeasible benchmark actions")
        return elapsed

    for _ in range(warmup):
        one_call()
    gc.collect()
    gc.disable()
    try:
        samples = [one_call() / 1_000_000 for _ in range(repeats)]
    finally:
        gc.enable()
        policy.reset_runtime()
    ordered = sorted(samples)
    median_ms = statistics.median(samples)
    p95_ms = ordered[max(0, int(0.95 * (len(ordered) - 1)))]
    return {
        "decisions_per_call": count,
        "network_microbatch_limit": 64,
        "warmup_calls": warmup,
        "timed_calls": repeats,
        "median_call_ms": median_ms,
        "p95_call_ms": p95_ms,
        "median_amortized_ms_per_decision": median_ms / count,
        "median_decisions_per_second": count * 1000.0 / median_ms,
        "min_call_ms": min(samples),
        "max_call_ms": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New directory; refuse to overwrite any prior run")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[1, 16, 64, 256])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if min(args.batch_sizes) < 1 or args.warmup < 1 or args.repeats < 3:
        raise ValueError("Invalid batch sizes, warmup, or repeats")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(1)
    checkpoint = args.checkpoint.resolve()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["signature"]["method"] != ProposedPacketPolicy.name:
        raise ValueError("Checkpoint is not the complete Proposed packet model")
    config = SimulationConfig()
    policy = ProposedPacketPolicy(config, saved["signature"]["seed"],
                                  training=False, device=args.device)
    policy.model.load_state_dict(saved["policy"]["model_state"])
    policy.model.eval()
    del saved
    gc.collect()
    snapshot = make_snapshot(config)
    results = []
    for count in args.batch_sizes:
        decisions = make_decisions(count, snapshot)
        result = benchmark(policy, snapshot, decisions, args.warmup,
                           args.repeats)
        results.append(result)
        print(json.dumps(result), flush=True)
    metadata = {
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "benchmark_scope": "whole ProposedPacketPolicy.route_packets call",
        "checkpoint": str(checkpoint),
        "device": args.device,
        "device_name": (torch.cuda.get_device_name(0) if args.device == "cuda"
                        else platform.processor()),
        "cpu_logical_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "packet_delay_includes_inference_runtime": False,
        "results": results,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "inference_benchmark.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("OUTPUT=" + str(output / "inference_benchmark.json"), flush=True)
    print("FINAL_JSON=" + json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
