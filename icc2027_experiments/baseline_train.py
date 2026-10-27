"""Resumable, episode-boundary training for adapted external baselines.

This runner is seed-11 precheck capable; it does not launch formal evaluation or
other seeds.  Checkpoint files contain full replay and optimizer state.
"""

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import statistics
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from .baseline_methods import BASELINE_NAMES, AdaptedBaselinePolicy
from .baseline_packet import PerPacketRoutingEnvironment, make_training_trace
from .config import SimulationConfig
from .proposed_packet import PACKET_ABLATION_VARIANTS, ProposedPacketPolicy


METHOD_NAMES = BASELINE_NAMES + (ProposedPacketPolicy.name,) + tuple(PACKET_ABLATION_VARIANTS)


FIELDS = (
    "episode", "traffic_seed", "trace_sha256", "reward_mean", "mean_loss",
    "delivery_ratio", "epsilon", "train_decisions", "total_train_decisions",
    "optimizer_updates", "total_optimizer_updates", "generated", "delivered",
    "dropped", "in_flight", "mean_queue", "max_queue", "wall_seconds",
)


def _atomic_save(value: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".partial")
    try:
        with temporary.open("wb") as stream:
            torch.save(value, stream)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_metrics(rows: list, path: Path) -> None:
    temporary = path.with_name(path.name + ".partial")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_checkpoint(path: Path, device: str) -> dict:
    # Keep replay and RNG tensors on CPU; model/optimizer restoration moves
    # trainable tensors to the selected device without exhausting VRAM.
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    with path.open("rb") as stream:
        return torch.load(stream, **options)


def _training_signature(method: str, seed: int, load_mbps: float, warmup_slots: int,
                        train_slots: int, batch_size: int, update_every_slots: int,
                        simulation: SimulationConfig,
                        max_train_decisions: Optional[int],
                        epsilon_decay_decisions: Optional[int]) -> dict:
    signature = {
        "method": method, "seed": seed, "load_mbps": load_mbps,
        "warmup_slots": warmup_slots, "train_slots": train_slots,
        "batch_size": batch_size, "simulation": asdict(simulation),
        "update_every_slots": update_every_slots,
        "trace_seed_rule": "seed*100000+17+episode*1009",
        "format_version": 1,
    }
    if max_train_decisions is not None:
        signature["max_train_decisions"] = max_train_decisions
        source_dir = Path(__file__).parent
        signature["source_sha256"] = {
            name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
            for name in ("baseline_train.py", "baseline_packet.py", "baseline_methods.py",
                         "proposed_packet.py", "models.py", "environment.py", "config.py")
        }
    if epsilon_decay_decisions is not None:
        signature["epsilon_decay_decisions"] = epsilon_decay_decisions
    return signature


def train(
    method: str, seed: int, load_mbps: float, episodes: int, output: Path,
    *, warmup_slots: int = 30, train_slots: int = 100,
    batch_size: Optional[int] = None, device: str = "cpu", resume: bool = False,
    update_every_slots: Optional[int] = None,
    max_train_decisions: Optional[int] = None,
    epsilon_decay_decisions: Optional[int] = None,
    checkpoint_every: int = 1,
    stop_after_episode: Optional[int] = None,
) -> dict:
    if method not in METHOD_NAMES:
        raise ValueError(f"Unknown method: {method}")
    if episodes < 1 or warmup_slots < 0 or train_slots < 1:
        raise ValueError("Invalid episode or slot count")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    if update_every_slots is not None and update_every_slots < 1:
        raise ValueError("update_every_slots must be positive")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    if epsilon_decay_decisions is not None and epsilon_decay_decisions < 1:
        raise ValueError("epsilon_decay_decisions must be positive")
    if stop_after_episode is not None and not 1 <= stop_after_episode <= episodes:
        raise ValueError("stop_after_episode must be within 1..episodes")
    simulation = SimulationConfig()
    max_episode_decisions = train_slots * simulation.num_satellites * simulation.service_packets_per_slot
    if max_train_decisions is not None and max_train_decisions < max_episode_decisions:
        raise ValueError("Decision budget is too small for one full episode")
    torch.set_num_threads(1)
    interval = update_every_slots or (10 if method == "graphpr_adapted" else 1)
    if method == ProposedPacketPolicy.name or method in PACKET_ABLATION_VARIANTS:
        policy = ProposedPacketPolicy(simulation, seed, batch_size=batch_size,
                                      device=device, update_every_slots=interval,
                                      epsilon_decay_decisions=epsilon_decay_decisions,
                                      variant=PACKET_ABLATION_VARIANTS.get(method, "gat_lstm_dqn"))
    else:
        policy = AdaptedBaselinePolicy(method, simulation, seed, pilot_batch_size=batch_size,
                                       device=device, update_every_slots=interval,
                                       epsilon_decay_decisions=epsilon_decay_decisions)
    signature = _training_signature(method, seed, load_mbps, warmup_slots,
                                    train_slots, policy.batch_size, interval, simulation,
                                    max_train_decisions, epsilon_decay_decisions)
    checkpoint = output / "checkpoint_latest.pt"
    metrics = output / "training_metrics.csv"
    complete = output / "training_complete.ok"
    rows = []
    next_episode = 0
    if resume:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"No resumable checkpoint: {checkpoint}")
        saved = _load_checkpoint(checkpoint, device)
        if saved["signature"] != signature:
            raise ValueError("Checkpoint training signature does not match requested run")
        next_episode = int(saved["next_episode"])
        if next_episode > episodes:
            raise ValueError("Requested target episodes is below checkpoint progress")
        rows = list(saved["rows"])
        if len(rows) != next_episode:
            raise ValueError("Checkpoint row count disagrees with next episode")
        policy.load_checkpoint_state(saved["policy"])
        _write_metrics(rows, metrics)
        if complete.exists() and next_episode < episodes:
            complete.unlink()
    else:
        if output.exists():
            raise FileExistsError(f"Training output already exists: {output}")
        output.mkdir(parents=True)
        (output / "training_config.json").write_text(
            json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    upper = min(episodes, stop_after_episode) if stop_after_episode else episodes
    budget_reached = False
    for episode in range(next_episode, upper):
        previous_decisions = int(rows[-1]["total_train_decisions"]) if rows else 0
        if (max_train_decisions is not None and
                previous_decisions + max_episode_decisions > max_train_decisions):
            budget_reached = True
            break
        started = time.perf_counter()
        policy.reset_runtime()
        policy.episode_reward_sum = 0.0
        policy.episode_reward_count = 0
        policy.losses.clear()
        traffic_seed = seed * 100_000 + 17 + episode * 1009
        trace = make_training_trace(simulation, load_mbps, traffic_seed,
                                    warmup_slots + train_slots)
        trace_hash = hashlib.sha256(repr(trace).encode("utf-8")).hexdigest()
        environment = PerPacketRoutingEnvironment(simulation, load_mbps, trace, policy)
        environment.measurement_active = False
        policy.training = False
        for _ in range(warmup_slots):
            environment.advance_background()
        environment.reset_measurement_counters()
        policy.training = True
        decisions_before = policy.decisions
        environment_decisions_before = environment.route_decisions
        updates_before = sum(policy.updates)
        queue_samples = []
        for _ in range(train_slots):
            queue_samples.append(environment.advance_background())
            if not math.isfinite(queue_samples[-1]):
                raise RuntimeError(f"Nonfinite queue at episode {episode}")
            if any(len(queue) > simulation.max_queue_packets for queue in environment.node_queues):
                raise RuntimeError(f"Queue overflow at episode {episode}")
            if len(queue_samples) % 10 == 0:
                print(json.dumps({"episode": episode, "slot_complete": len(queue_samples),
                                  "train_slots": train_slots,
                                  "optimizer_updates_total": sum(policy.updates)}), flush=True)
        episode_decisions = policy.decisions - decisions_before
        if episode_decisions != environment.route_decisions - environment_decisions_before:
            raise RuntimeError("Policy/environment route decision mismatch")
        generated = environment.background_generated
        delivered = environment.background_delivered
        dropped = environment.background_dropped
        in_flight = sum(packet[3] for queue in environment.node_queues for packet in queue)
        if generated != delivered + dropped + in_flight:
            raise RuntimeError(f"Packet conservation failure at episode {episode}: "
                               f"{generated} != {delivered}+{dropped}+{in_flight}")
        if policy.episode_reward_count != episode_decisions:
            raise RuntimeError(f"Reward/decision count mismatch at episode {episode}")
        if not all(math.isfinite(loss) for loss in policy.losses):
            raise RuntimeError(f"Nonfinite loss at episode {episode}")
        mean_loss = statistics.mean(policy.losses) if policy.losses else None
        reward_mean = (policy.episode_reward_sum / policy.episode_reward_count
                       if policy.episode_reward_count else None)
        epsilon_used = policy.epsilon
        total_decisions = previous_decisions + episode_decisions
        policy.finish_episode(total_decisions)
        row = {
            "episode": episode, "traffic_seed": traffic_seed, "trace_sha256": trace_hash,
            "reward_mean": reward_mean, "mean_loss": mean_loss,
            "delivery_ratio": delivered / generated if generated else None,
            "epsilon": epsilon_used, "train_decisions": episode_decisions,
            "total_train_decisions": total_decisions,
            "optimizer_updates": sum(policy.updates) - updates_before,
            "total_optimizer_updates": sum(policy.updates),
            "generated": generated, "delivered": delivered, "dropped": dropped,
            "in_flight": in_flight, "mean_queue": statistics.mean(queue_samples),
            "max_queue": max(len(queue) for queue in environment.node_queues),
            "wall_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        policy.losses.clear()
        next_exceeds_budget = (max_train_decisions is not None and
                               total_decisions + max_episode_decisions > max_train_decisions)
        if ((episode + 1) % checkpoint_every == 0 or episode + 1 == upper or
                next_exceeds_budget):
            saved = {"signature": signature, "next_episode": episode + 1,
                     "rows": rows, "policy": policy.checkpoint_state()}
            _atomic_save(saved, checkpoint)
        _write_metrics(rows, metrics)
        failure = output / "training_failed.txt"
        if failure.exists():
            failure.unlink()
        print(json.dumps({key: row[key] for key in
                          ("episode", "reward_mean", "mean_loss", "delivery_ratio",
                           "epsilon", "train_decisions", "optimizer_updates", "wall_seconds")}),
              flush=True)
    total_decisions = int(rows[-1]["total_train_decisions"]) if rows else 0
    if max_train_decisions is not None and total_decisions + max_episode_decisions > max_train_decisions:
        budget_reached = True
    finished = len(rows) >= episodes or budget_reached
    if finished:
        complete.write_text(
            f"episodes={len(rows)}\ntrain_decisions={total_decisions}\n"
            f"budget_reached={budget_reached}\n", encoding="utf-8"
        )
    return {"completed_episodes": len(rows), "target_episodes": episodes,
            "total_train_decisions": total_decisions,
            "checkpoint": str(checkpoint), "complete": finished,
            "budget_reached": budget_reached}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHOD_NAMES, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--load-mbps", type=float, default=240.0)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--warmup-slots", type=int, default=30)
    parser.add_argument("--train-slots", type=int, default=100)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--update-every-slots", type=int)
    parser.add_argument("--max-train-decisions", type=int)
    parser.add_argument("--epsilon-decay-decisions", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = train(args.method, args.seed, args.load_mbps, args.episodes,
                       args.output_dir, warmup_slots=args.warmup_slots,
                       train_slots=args.train_slots, batch_size=args.batch_size,
                       device=args.device, resume=args.resume,
                       update_every_slots=args.update_every_slots,
                       max_train_decisions=args.max_train_decisions,
                       epsilon_decay_decisions=args.epsilon_decay_decisions,
                       checkpoint_every=args.checkpoint_every)
    except Exception:
        if args.output_dir.is_dir() and not (args.output_dir / "training_complete.ok").exists():
            (args.output_dir / "training_failed.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        raise
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
