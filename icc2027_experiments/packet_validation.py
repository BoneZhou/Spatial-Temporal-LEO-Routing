"""Independent, paired traffic validation of frozen packet-routing policies.

This is a seed-11 validation check, not formal multi-seed paper statistics.
Only inference weights are copied from the full resumable training checkpoint.
"""

import argparse
import gc
import hashlib
import json
import math
import statistics
import inspect
from dataclasses import asdict
from pathlib import Path

import torch

from .baseline_methods import BASELINE_NAMES, AdaptedBaselinePolicy
from .baseline_packet import PerPacketRoutingEnvironment, make_training_trace
from .config import SimulationConfig
from .proposed_packet import ProposedPacketPolicy


METHODS = (ProposedPacketPolicy.name,) + BASELINE_NAMES
PROJECT = Path(__file__).resolve().parent.parent
TRAINING_ROOT = PROJECT / "outputs" / "packet_fair_seed11_load240_budget10m"
VALIDATION_SEEDS = (601, 602, 603)


def validate_evaluation_metrics(result: dict) -> None:
    """Reject numerical failures while allowing undefined delay with no deliveries."""
    for field in ("throughput_mbps", "packet_loss_rate", "average_queue_packets"):
        value = result[field]
        if value is None or not math.isfinite(value):
            raise RuntimeError(
                f"Non-finite validation metric {field}: "
                f"{result['method']}, {result['traffic_seed']}"
            )
    delay = result["end_to_end_delay_ms"]
    delivered = int(result["delivered_packets"])
    if delivered == 0:
        if delay is not None:
            raise RuntimeError("Zero-delivery validation must report undefined delay")
    elif delay is None or not math.isfinite(delay):
        raise RuntimeError(
            "Non-finite validation metric end_to_end_delay_ms: "
            f"{result['method']}, {result['traffic_seed']}"
        )


def load_frozen_policy(method: str, device: str, seed: int = 11):
    """Load model weights, without restoring optimizer, replay or exploration."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    method_dir = (TRAINING_ROOT if seed == 11 else
                  PROJECT / "outputs" / "packet_fair_multiseed_load240_budget10m" /
                  f"seed{seed}") / method
    checkpoint = method_dir / "checkpoint_latest.pt"
    if not (method_dir / "training_complete.ok").is_file():
        raise RuntimeError(f"Training is not marked complete: {method}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    # CPU fallback on the installed Torch 1.13 runtime has no mmap option.
    load_options = {"mmap": True} if "mmap" in inspect.signature(torch.load).parameters else {}
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, **load_options)
    signature = saved["signature"]
    if (signature["method"] != method or signature["seed"] != seed or
            signature["load_mbps"] != 240.0 or
            signature["max_train_decisions"] != 10_000_000):
        raise ValueError(f"Unexpected training signature: {method}")
    simulation = SimulationConfig()
    if signature["simulation"] != asdict(simulation):
        raise ValueError(f"Physical configuration mismatch: {method}")
    if method == ProposedPacketPolicy.name:
        policy = ProposedPacketPolicy(simulation, seed, training=False, device=device)
        policy.model.load_state_dict(saved["policy"]["model_state"])
        policy.model.eval()
    else:
        policy = AdaptedBaselinePolicy(method, simulation, seed, training=False,
                                       device=device)
        states = saved["policy"]["model_states"]
        if len(states) != policy.agent_count:
            raise ValueError(f"Agent count mismatch: {method}")
        for model, state in zip(policy.models, states):
            model.load_state_dict(state)
            model.eval()
    if policy.epsilon != 0.0 or policy.training:
        raise RuntimeError("Validation must be greedy and inference-only")
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "train_decisions": int(saved["rows"][-1]["total_train_decisions"]),
        "train_episodes": int(saved["next_episode"]),
        "training_signature": signature,
        "evaluation_torch_version": torch.__version__,
        "evaluation_device": device,
    }
    if method == "fdr_marl_adapted":
        state = saved["policy"]
        metadata["training_sanity"] = {
            "replay_sizes": [len(r) for r in state["replays"]],
            "optimizer_updates_per_agent": state["updates"],
            "epsilon_first": saved["rows"][0]["epsilon"],
            "epsilon_last": state["epsilon"],
            "total_optimizer_updates": saved["rows"][-1]["total_optimizer_updates"],
            "target_syncs_expected_per_agent": [n//500 for n in state["updates"]],
            "optimizer_step_max_per_agent": [max(
                (float(v["step"]) for v in optimizer["state"].values()), default=0)
                for optimizer in state["optimizer_states"]],
        }
    del saved
    gc.collect()
    return policy, simulation, metadata


def evaluate_one(policy, simulation: SimulationConfig, method: str, seed: int,
                 warmup_slots: int, measurement_slots: int, metadata: dict) -> dict:
    if seed in (101, 202, 303, 404, 505):
        raise ValueError("Reserved formal test traffic seed cannot be used for validation")
    if warmup_slots < 1 or measurement_slots < 1:
        raise ValueError("Warmup and measurement windows must be positive")
    if policy.training or policy.epsilon != 0.0:
        raise RuntimeError("Policy is not frozen for validation")
    load_mbps = 240.0
    trace = make_training_trace(simulation, load_mbps, seed,
                                warmup_slots + measurement_slots)
    trace_hash = hashlib.sha256(repr(trace).encode("utf-8")).hexdigest()
    policy.reset_runtime()
    environment = PerPacketRoutingEnvironment(simulation, load_mbps, trace, policy)
    environment.measurement_active = False
    for _ in range(warmup_slots):
        environment.advance_background()
    environment.reset_measurement_counters()
    decisions_before = policy.decisions
    environment_decisions_before = environment.route_decisions
    queue_samples = []
    for _ in range(measurement_slots):
        queue = environment.advance_background()
        if not math.isfinite(queue) or not 0 <= queue <= simulation.max_queue_packets:
            raise RuntimeError(f"Invalid average queue: {method}, traffic seed {seed}")
        if any(len(row) > simulation.max_queue_packets for row in environment.node_queues):
            raise RuntimeError(f"Queue overflow: {method}, traffic seed {seed}")
        queue_samples.append(queue)
    generated = environment.background_generated
    delivered = environment.background_delivered
    dropped = environment.background_dropped
    in_flight = sum(packet[3] for queue in environment.node_queues for packet in queue)
    if generated != delivered + dropped + in_flight:
        raise RuntimeError(f"Packet conservation failed: {method}, traffic seed {seed}")
    if dropped != (environment.background_queue_dropped +
                   environment.background_ttl_dropped +
                   environment.background_no_route_dropped):
        raise RuntimeError(f"Drop reason accounting failed: {method}, traffic seed {seed}")
    decisions = policy.decisions - decisions_before
    if decisions != environment.route_decisions - environment_decisions_before:
        raise RuntimeError(f"Decision accounting failed: {method}, traffic seed {seed}")
    if len(environment.delivered_delay_slots) != delivered:
        raise RuntimeError(f"Delay accounting failed: {method}, traffic seed {seed}")
    metrics = environment.background_metrics(measurement_slots)
    result = {
        "status": "VALIDATION_ONLY",
        "method": method,
        "checkpoint_seed": 11,
        "traffic_seed": seed,
        "trace_sha256": trace_hash,
        "load_mbps": load_mbps,
        "warmup_slots": warmup_slots,
        "measurement_slots": measurement_slots,
        "epsilon": policy.epsilon,
        "generated_packets": generated,
        "delivered_packets": delivered,
        "dropped_packets": dropped,
        "queue_dropped_packets": environment.background_queue_dropped,
        "ttl_dropped_packets": environment.background_ttl_dropped,
        "no_route_dropped_packets": environment.background_no_route_dropped,
        "in_flight_at_horizon": in_flight,
        "route_decisions": decisions,
        "throughput_mbps": metrics["throughput_mbps"],
        "packet_loss_rate": metrics["packet_loss"],
        "end_to_end_delay_ms": (
            statistics.mean(environment.delivered_delay_slots) *
            simulation.slot_duration_s * 1000.0 if delivered else None
        ),
        "delay_defined": delivered > 0,
        "average_queue_packets": statistics.mean(queue_samples),
        "max_queue_packets": max(len(queue) for queue in environment.node_queues),
        "checkpoint": metadata["checkpoint"],
        "checkpoint_bytes": metadata["checkpoint_bytes"],
        "train_decisions": metadata["train_decisions"],
        "train_episodes": metadata["train_episodes"],
        "loss_definition": "Dropped measured packets / generated measured packets; in-flight packets are separate, not loss.",
        "delay_definition": "Mean birth-to-delivery slot sojourn of delivered measured packets, times 20 ms.",
        "queue_definition": "Mean of per-slot, per-satellite shared-queue lengths in the measurement window.",
    }
    validate_evaluation_metrics(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--traffic-seeds", type=int, nargs="+", default=list(VALIDATION_SEEDS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmup-slots", type=int, default=30)
    parser.add_argument("--measurement-slots", type=int, default=100)
    args = parser.parse_args()
    if len(set(args.traffic_seeds)) != len(args.traffic_seeds):
        raise ValueError("Duplicate traffic seeds")
    torch.set_num_threads(1)
    policy, simulation, metadata = load_frozen_policy(args.method, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.traffic_seeds:
        path = args.output_dir / f"{args.method}_traffic{seed}_VALIDATION.json"
        if path.exists():
            prior = json.loads(path.read_text(encoding="utf-8"))
            if (prior.get("status") != "VALIDATION_ONLY" or
                    prior.get("method") != args.method or
                    prior.get("traffic_seed") != seed or
                    prior.get("checkpoint") != metadata["checkpoint"] or
                    prior.get("checkpoint_bytes") != metadata["checkpoint_bytes"] or
                    prior.get("warmup_slots") != args.warmup_slots or
                    prior.get("measurement_slots") != args.measurement_slots):
                raise RuntimeError(f"Incompatible validation output: {path}")
            print(json.dumps({"method": args.method, "traffic_seed": seed,
                              "status": "already_complete"}), flush=True)
            continue
        result = evaluate_one(policy, simulation, args.method, seed, args.warmup_slots,
                              args.measurement_slots, metadata)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({
            "method": args.method, "traffic_seed": seed,
            "generated": result["generated_packets"],
            "delivered": result["delivered_packets"],
            "dropped": result["dropped_packets"],
            "in_flight": result["in_flight_at_horizon"],
        }), flush=True)


if __name__ == "__main__":
    main()
