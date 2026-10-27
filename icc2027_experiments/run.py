import argparse
import csv
import json
import math
import random
import statistics
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .config import ExperimentConfig, build_config
from .environment import LEORoutingEnvironment, RoutingRequest
from .models import VARIANTS, UnifiedRoutingQNetwork


Transition = Tuple[torch.Tensor, int, float, torch.Tensor, bool, int, int, int]
CUDA_BACKGROUND_BATCH_SIZE = 256


def resolve_variant_device(
    variant: str,
    gat_lstm_device: str,
    gat_device: Optional[str] = None,
    device: Optional[str] = None,
) -> torch.device:
    """Resolve generic and backward-compatible variant-specific device options."""
    if device is not None:
        requested_device = device
    elif variant not in ("gat_dqn", "gat_lstm_dqn"):
        return torch.device("cpu")
    else:
        requested_device = (
            gat_device
            if gat_device is not None
            else (gat_lstm_device if variant == "gat_lstm_dqn" else "cpu")
        )
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this PyTorch installation "
                "cannot access CUDA"
            )
        return torch.device("cuda")
    raise ValueError(f"Unsupported device: {requested_device}")


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def atomic_torch_save(payload: Dict[str, object], path: Path) -> None:
    """Atomically replace a checkpoint so interruptions cannot leave a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def validate_resume_state(
    state: Dict[str, Any],
    variant: str,
    config: ExperimentConfig,
) -> None:
    if state.get("checkpoint_version") != 3:
        raise ValueError("Unsupported or incomplete training checkpoint")
    if state.get("variant") != variant:
        raise ValueError(
            f"Checkpoint variant {state.get('variant')!r} does not match {variant!r}"
        )

    saved_config = dict(state["config"])
    current_config = config.to_dict()
    saved_learning = dict(saved_config["learning"])
    current_learning = dict(current_config["learning"])
    for field in ("training_episodes", "evaluation_episodes"):
        saved_learning.pop(field, None)
        current_learning.pop(field, None)
    saved_config["learning"] = saved_learning
    current_config["learning"] = current_learning
    if saved_config != current_config:
        raise ValueError("Checkpoint configuration does not match the requested run")

    next_episode = int(state["next_episode"])
    if next_episode > config.learning.training_episodes:
        raise ValueError(
            "Checkpoint is already beyond the requested number of training episodes"
        )


def masked_argmax(q_values: torch.Tensor, feasible_actions: Sequence[int]) -> int:
    if not feasible_actions:
        raise ValueError("No feasible action is available")
    mask = torch.full_like(q_values, -1e9)
    mask[list(feasible_actions)] = 0.0
    return int(torch.argmax(q_values + mask).item())


def state_history(initial_state: torch.Tensor, window: int) -> Deque[torch.Tensor]:
    return deque([initial_state.clone() for _ in range(window)], maxlen=window)


def stack_history(history: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(history), dim=0)


class ModelBackgroundRouter:
    """Batched policy adapter used to route every background traffic group."""

    def __init__(
        self,
        model: UnifiedRoutingQNetwork,
        env: LEORoutingEnvironment,
        observation_window: int,
        epsilon: float,
        rng: random.Random,
    ):
        self.model = model
        self.env = env
        self.observation_window = observation_window
        self.epsilon = epsilon
        self.rng = rng
        self.histories: Dict[RoutingRequest, Deque[torch.Tensor]] = {}

    def __call__(
        self,
        requests: Sequence[RoutingRequest],
        step_index: int,
        neighbors: List[List[int]],
    ) -> Dict[RoutingRequest, int]:
        if not requests:
            return {}
        sequences = []
        current_states = self.env.state_batch(requests, step_index)
        for request, state in zip(requests, current_states):
            if request not in self.histories:
                self.histories[request] = state_history(state, self.observation_window)
            else:
                self.histories[request].append(state)
            sequences.append(stack_history(self.histories[request]))

        device = model_device(self.model)
        adjacency = self.env.adjacency_tensor(step_index).to(device)
        with torch.no_grad():
            if device.type == "cuda":
                # A congested slot can contain thousands of background routing
                # requests.  Running them as one dense-GAT batch can consume nearly
                # all VRAM, so keep histories/replay on CPU and stream bounded
                # inference batches through the GPU.
                q_value_batches = []
                for start in range(0, len(sequences), CUDA_BACKGROUND_BATCH_SIZE):
                    state_batch = torch.stack(
                        sequences[start : start + CUDA_BACKGROUND_BATCH_SIZE]
                    ).to(device)
                    q_value_batches.append(
                        self.model(state_batch, adjacency).cpu()
                    )
                q_values = torch.cat(q_value_batches, dim=0)
            else:
                state_batch = torch.stack(sequences)
                q_values = self.model(state_batch, adjacency)

        actions: Dict[RoutingRequest, int] = {}
        for index, request in enumerate(requests):
            source, _ = request
            feasible = neighbors[source]
            if not feasible:
                continue
            if self.rng.random() < self.epsilon:
                actions[request] = self.rng.choice(feasible)
            else:
                actions[request] = masked_argmax(q_values[index], feasible)
        return actions


def choose_action(
    model: UnifiedRoutingQNetwork,
    state_sequence: torch.Tensor,
    adjacency: torch.Tensor,
    feasible: Sequence[int],
    epsilon: float,
    rng: random.Random,
) -> int:
    if rng.random() < epsilon:
        return rng.choice(list(feasible))
    device = model_device(model)
    with torch.no_grad():
        q_values = model(
            state_sequence.unsqueeze(0).to(device),
            adjacency.to(device),
        )[0]
    return masked_argmax(q_values, feasible)


def update_network(
    model: UnifiedRoutingQNetwork,
    target: UnifiedRoutingQNetwork,
    optimizer: torch.optim.Optimizer,
    replay: Deque[Transition],
    env: LEORoutingEnvironment,
    config: ExperimentConfig,
) -> float:
    learning = config.learning
    device = model_device(model)
    batch = random.sample(replay, learning.batch_size)
    states, actions, rewards, next_states, dones, steps, next_steps, next_nodes = zip(*batch)
    state_batch = torch.stack(states).to(device)
    next_state_batch = torch.stack(next_states).to(device)
    adjacency = torch.stack([env.adjacency_tensor(step) for step in steps]).to(device)
    next_adjacency = torch.stack(
        [env.adjacency_tensor(step) for step in next_steps]
    ).to(device)

    q_values = model(state_batch, adjacency)
    action_tensor = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    selected_q = q_values.gather(1, action_tensor).squeeze(1)

    with torch.no_grad():
        next_q_all = target(next_state_batch, next_adjacency)
        next_q_values = []
        for index, (step, node, done) in enumerate(zip(next_steps, next_nodes, dones)):
            if done:
                next_q_values.append(torch.tensor(0.0, device=device))
            else:
                feasible = env.neighbors_at(step)[node]
                next_q_values.append(next_q_all[index, feasible].max())
        next_q = torch.stack(next_q_values)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        target_q = reward_tensor + learning.discount_factor * next_q

    loss = nn.functional.smooth_l1_loss(selected_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def train_variant(
    variant: str,
    config: ExperimentConfig,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 0,
    resume_state: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[UnifiedRoutingQNetwork, List[Dict[str, object]]]:
    device = device or torch.device("cpu")
    torch.manual_seed(config.model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.model_seed)
    random.seed(config.model_seed)
    env = LEORoutingEnvironment(config.simulation, config.offered_load_mbps)
    learning = config.learning
    model = UnifiedRoutingQNetwork(
        variant, 6, learning.hidden_dim, learning.lstm_hidden_dim
    ).to(device)
    target = UnifiedRoutingQNetwork(
        variant, 6, learning.hidden_dim, learning.lstm_hidden_dim
    ).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning.learning_rate)
    replay: Deque[Transition] = deque(maxlen=learning.replay_buffer_size)
    action_rng = random.Random(config.model_seed + 9_001)
    epsilon = learning.epsilon_start
    updates = 0
    training_rows: List[Dict[str, object]] = []
    start_episode = 0

    if resume_state is not None:
        validate_resume_state(resume_state, variant, config)
        model.load_state_dict(resume_state["state_dict"])
        target.load_state_dict(resume_state["target_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        replay.extend(resume_state["replay"])
        epsilon = float(resume_state["epsilon"])
        updates = int(resume_state["updates"])
        training_rows = list(resume_state["training_rows"])
        start_episode = int(resume_state["next_episode"])
        random.setstate(resume_state["python_random_state"])
        torch.set_rng_state(resume_state["torch_rng_state"])
        if (
            device.type == "cuda"
            and resume_state.get("cuda_rng_state_all") is not None
        ):
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state_all"])
        action_rng.setstate(resume_state["action_rng_state"])

    model.train()
    for episode in range(start_episode, learning.training_episodes):
        epsilon_start = epsilon
        env.reset(config.traffic_seed + episode)
        current, destination = env.sample_source_destination()
        background_router = ModelBackgroundRouter(
            model,
            env,
            config.observation_window,
            0.0,
            random.Random(config.model_seed + 50_000 + episode),
        )
        for _ in range(config.simulation.training_warmup_slots):
            env.advance_background(background_router)
        pre_focal_queue = env.average_node_queue()
        pre_focal_full_queues = sum(
            len(queue) >= config.simulation.max_queue_packets
            for queue in env.node_queues
        )
        initial = env.state_tensor(current, destination, env.step_index)
        history = state_history(initial, config.observation_window)
        visited = {current}
        total_reward = 0.0
        episode_losses: List[float] = []
        queue_samples: List[float] = []
        delivered = False
        dropped = False
        hops = 0

        for hop in range(config.simulation.max_hops):
            step = env.step_index
            adjacency = env.adjacency_tensor(step)
            feasible = [node for node in env.neighbors_at(step)[current] if node not in visited]
            if not feasible:
                feasible = env.neighbors_at(step)[current]
            sequence = stack_history(history)
            action = choose_action(model, sequence, adjacency, feasible, epsilon, action_rng)
            result = env.step(
                current,
                action,
                destination,
                hop,
                background_router=background_router,
            )
            hops = hop + 1
            total_reward += result.reward
            queue_samples.append(result.average_queue)
            delivered = result.delivered
            dropped = result.dropped
            next_state = env.state_tensor(result.next_node, destination, env.step_index)
            next_history = deque(history, maxlen=config.observation_window)
            next_history.append(next_state)
            done = result.delivered or result.dropped
            replay.append(
                (
                    sequence.detach(),
                    action,
                    result.reward,
                    stack_history(next_history).detach(),
                    done,
                    step,
                    env.step_index,
                    result.next_node,
                )
            )
            if len(replay) >= learning.batch_size:
                episode_losses.append(
                    update_network(model, target, optimizer, replay, env, config)
                )
                updates += 1
                if updates % learning.target_update_steps == 0:
                    target.load_state_dict(model.state_dict())
            history = next_history
            current = result.next_node
            visited.add(current)
            if done:
                break
        epsilon = max(learning.epsilon_end, epsilon * learning.epsilon_decay)
        row: Dict[str, object] = {
            "phase": "training",
            "variant": variant,
            "seed": config.seed,
            "model_seed": config.model_seed,
            "traffic_seed": config.traffic_seed + episode,
            "offered_load_mbps": config.offered_load_mbps,
            "observation_window": config.observation_window,
            "episode": episode,
            "background_policy_epsilon": 0.0,
            "pre_focal_warmup_slots": config.simulation.training_warmup_slots,
            "pre_focal_average_queue_packets": round(pre_focal_queue, 6),
            "pre_focal_full_queue_nodes": pre_focal_full_queues,
            "total_reward": round(total_reward, 6),
            "mean_loss": (
                round(statistics.mean(episode_losses), 6) if episode_losses else ""
            ),
            "delivery_ratio": float(delivered),
            "delivered": int(delivered),
            "dropped": int(dropped or not delivered),
            "hops": hops,
            "average_queue_packets": (
                round(statistics.mean(queue_samples), 6) if queue_samples else 0.0
            ),
            "epsilon_start": round(epsilon_start, 8),
            "epsilon_end": round(epsilon, 8),
            "optimizer_updates": len(episode_losses),
        }
        training_rows.append(row)
        if progress_callback is not None:
            progress_callback(row)
        completed_episodes = episode + 1
        if (
            checkpoint_path is not None
            and checkpoint_every > 0
            and (
                completed_episodes % checkpoint_every == 0
                or completed_episodes == learning.training_episodes
            )
        ):
            atomic_torch_save(
                {
                    "checkpoint_version": 3,
                    "variant": variant,
                    "config": config.to_dict(),
                    "next_episode": completed_episodes,
                    "epsilon": epsilon,
                    "updates": updates,
                    "state_dict": model.state_dict(),
                    "target_state_dict": target.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "replay": list(replay),
                    "python_random_state": random.getstate(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all()
                        if device.type == "cuda"
                        else None
                    ),
                    "action_rng_state": action_rng.getstate(),
                    "training_rows": training_rows,
                },
                checkpoint_path,
            )
    return model, training_rows


def evaluate_variant(
    variant: str,
    model: UnifiedRoutingQNetwork,
    config: ExperimentConfig,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    start_episode: int = 0,
) -> List[Dict[str, object]]:
    env = LEORoutingEnvironment(config.simulation, config.offered_load_mbps)
    rows: List[Dict[str, object]] = []
    model.eval()
    for episode in range(start_episode, config.learning.evaluation_episodes):
        episode_seed = config.traffic_seed + 1_000_000 + episode
        env.reset(episode_seed)
        current, destination = env.sample_source_destination()
        source_node = current
        background_router = ModelBackgroundRouter(
            model,
            env,
            config.observation_window,
            0.0,
            random.Random(config.model_seed + 2_000_000 + episode),
        )
        for _ in range(config.simulation.evaluation_warmup_slots):
            env.advance_background(background_router)
        pre_focal_queue = env.average_node_queue()
        pre_focal_full_queues = sum(
            len(queue) >= config.simulation.max_queue_packets
            for queue in env.node_queues
        )
        env.reset_measurement_counters()
        measurement_start = env.step_index
        measurement_end = measurement_start + config.simulation.evaluation_measurement_slots
        history = state_history(
            env.state_tensor(current, destination, env.step_index),
            config.observation_window,
        )
        visited = {current}
        total_reward = 0.0
        total_delay = 0.0
        propagation = 0.0
        transmission = 0.0
        queueing = 0.0
        processing = 0.0
        queue_samples: List[float] = []
        delivered = False
        dropped = False
        hops = 0

        for hop in range(config.simulation.max_hops):
            if env.step_index >= measurement_end:
                dropped = True
                break
            step = env.step_index
            feasible = [node for node in env.neighbors_at(step)[current] if node not in visited]
            if not feasible:
                feasible = env.neighbors_at(step)[current]
            action = choose_action(
                model,
                stack_history(history),
                env.adjacency_tensor(step),
                feasible,
                0.0,
                random.Random(episode_seed),
            )
            result = env.step(
                current,
                action,
                destination,
                hop,
                background_router=background_router,
                max_simulated_slots=measurement_end - env.step_index,
            )
            hops = hop + 1
            total_reward += result.reward
            total_delay += result.delay_ms
            propagation += result.propagation_delay_ms
            transmission += result.transmission_delay_ms
            queueing += result.queueing_delay_ms
            processing += result.processing_delay_ms
            queue_samples.append(result.average_queue)
            next_state = env.state_tensor(result.next_node, destination, env.step_index)
            history.append(next_state)
            current = result.next_node
            visited.add(current)
            delivered = result.delivered
            dropped = result.dropped
            if delivered or dropped:
                break

        if not delivered and not dropped:
            dropped = True

        while env.step_index < measurement_end:
            queue_samples.append(env.advance_background(background_router))

        measured_slots = env.step_index - measurement_start
        background = env.background_metrics(measured_slots)

        row: Dict[str, object] = {
                "phase": (
                    "evaluation" if config.mode == "full" else f"{config.mode}_evaluation"
                ),
                "variant": variant,
                "seed": config.seed,
                "model_seed": config.model_seed,
                "traffic_seed": episode_seed,
                "offered_load_mbps": config.offered_load_mbps,
                "observation_window": config.observation_window,
                "episode": episode,
                "background_policy_epsilon": 0.0,
                "pre_focal_warmup_slots": config.simulation.evaluation_warmup_slots,
                "pre_focal_average_queue_packets": round(pre_focal_queue, 6),
                "pre_focal_full_queue_nodes": pre_focal_full_queues,
                "source_node": source_node,
                "destination_node": destination,
                "delivered": int(delivered),
                "focal_packet_loss": int(not delivered),
                "throughput_mbps": round(background["throughput_mbps"], 6),
                "packet_loss": round(background["packet_loss"], 6),
                "end_to_end_delay_ms": round(total_delay, 6) if delivered else "",
                "propagation_delay_ms": round(propagation, 6) if delivered else "",
                "transmission_delay_ms": round(transmission, 6) if delivered else "",
                "queueing_delay_ms": round(queueing, 6) if delivered else "",
                "processing_delay_ms": round(processing, 6) if delivered else "",
                "average_queue_packets": round(statistics.mean(queue_samples), 6) if queue_samples else 0.0,
                "hops": hops,
                "measured_slots": measured_slots,
                "background_generated": int(background["background_generated"]),
                "background_delivered": int(background["background_delivered"]),
                "background_dropped": int(background["background_dropped"]),
                "background_queue_dropped": int(background["background_queue_dropped"]),
                "background_ttl_dropped": int(background["background_ttl_dropped"]),
                "background_no_route_dropped": int(background["background_no_route_dropped"]),
                "total_reward": round(total_reward, 6),
            }
        rows.append(row)
        if progress_callback is not None:
            progress_callback(row)
    return rows


def numeric_values(rows: List[Dict[str, object]], field: str) -> List[float]:
    return [float(row[field]) for row in rows if row[field] != ""]


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(rows: List[Dict[str, object]], variant: str, config: ExperimentConfig) -> Dict[str, object]:
    result: Dict[str, object] = {
        "variant": variant,
        "seed": config.seed,
        "offered_load_mbps": config.offered_load_mbps,
        "observation_window": config.observation_window,
        "evaluation_episodes": len(rows),
        "status": {
            "smoke": "SMOKE_TEST_ONLY",
            "pilot": "PILOT_ONLY",
            "full": "FULL_RUN",
        }[config.mode],
    }
    for field in (
        "throughput_mbps",
        "packet_loss",
        "end_to_end_delay_ms",
        "pre_focal_average_queue_packets",
        "pre_focal_full_queue_nodes",
        "average_queue_packets",
        "total_reward",
    ):
        values = numeric_values(rows, field)
        mean, std = mean_std(values)
        result[f"{field}_mean"] = round(mean, 6) if math.isfinite(mean) else ""
        result[f"{field}_std"] = round(std, 6) if math.isfinite(std) else ""
    result["delivery_ratio"] = round(statistics.mean(numeric_values(rows, "delivered")), 6)
    return result


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def load_completed_evaluation_rows(
    path: Path, variant: str, config: ExperimentConfig
) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) > config.learning.evaluation_episodes:
        raise ValueError("Existing evaluation has more episodes than requested")
    for episode, row in enumerate(rows):
        expected = {
            "phase": "evaluation",
            "variant": variant,
            "seed": str(config.seed),
            "model_seed": str(config.model_seed),
            "traffic_seed": str(config.traffic_seed + 1_000_000 + episode),
            "offered_load_mbps": str(config.offered_load_mbps),
            "observation_window": str(config.observation_window),
            "episode": str(episode),
            "pre_focal_warmup_slots": str(config.simulation.evaluation_warmup_slots),
            "measured_slots": str(config.simulation.evaluation_measurement_slots),
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(
                    f"Existing evaluation row {episode} has unexpected {field}"
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled ICC 2027 routing ablations")
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--load-mbps", type=float, default=180.0)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--training-episodes", type=int, default=None)
    parser.add_argument("--evaluation-episodes", type=int, default=None)
    parser.add_argument(
        "--warmup-slots",
        type=int,
        default=None,
        help="Use the same number of background warm-up slots before training and evaluation focal packets",
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--gat-lstm-device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help=(
            "Execution device for gat_lstm_dqn only; all other variants always "
            "run on CPU"
        ),
    )
    parser.add_argument(
        "--gat-device",
        choices=("cpu", "cuda", "auto"),
        default=None,
        help=(
            "Execution device for both gat_dqn and gat_lstm_dqn. When omitted, "
            "the legacy --gat-lstm-device setting remains in effect only for "
            "gat_lstm_dqn."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default=None,
        help=(
            "Execution device for every requested variant. This overrides the "
            "backward-compatible GAT-specific device options."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save a resumable training checkpoint every N episodes; zero disables it",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume one variant from a full training checkpoint",
    )
    parser.add_argument(
        "--resume-evaluation",
        action="store_true",
        help="Continue incomplete evaluation episodes in the existing output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be at least one")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every cannot be negative")
    if args.warmup_slots is not None and args.warmup_slots < 0:
        raise ValueError("--warmup-slots cannot be negative")
    if args.resume_checkpoint is not None and len(args.variants) != 1:
        raise ValueError("--resume-checkpoint requires exactly one --variants value")
    if args.resume_evaluation and args.resume_checkpoint is None:
        raise ValueError("--resume-evaluation requires --resume-checkpoint")
    torch.set_num_threads(args.torch_threads)
    config = build_config(args.seed, args.load_mbps, args.window, args.mode)
    if args.warmup_slots is not None:
        config = replace(
            config,
            simulation=replace(
                config.simulation,
                evaluation_warmup_slots=args.warmup_slots,
            ),
        )
    if args.training_episodes is not None or args.evaluation_episodes is not None:
        learning = replace(
            config.learning,
            training_episodes=(
                args.training_episodes
                if args.training_episodes is not None
                else config.learning.training_episodes
            ),
            evaluation_episodes=(
                args.evaluation_episodes
                if args.evaluation_episodes is not None
                else config.learning.evaluation_episodes
            ),
        )
        config = replace(config, learning=learning)
    output_dir = args.output_dir or Path("outputs") / (
        f"{args.mode}_seed{args.seed}_load{int(args.load_mbps)}_tw{args.window}"
    )
    device_plan = {
        variant: resolve_variant_device(
            variant,
            args.gat_lstm_device,
            args.gat_device,
            args.device,
        )
        for variant in args.variants
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "resolved_config.json"
    if args.resume_evaluation and config_path.exists():
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        if saved_config != config.to_dict():
            raise ValueError("Existing evaluation configuration does not match")
    elif args.resume_evaluation and (output_dir / "raw_metrics.csv").exists():
        raise ValueError("Existing evaluation rows have no configuration file")
    config_path.write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    runtime_metadata = {
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "requested_gat_lstm_device": args.gat_lstm_device,
        "requested_gat_device": args.gat_device,
        "requested_device": args.device,
        "variant_devices": {
            variant: device.type for variant, device in device_plan.items()
        },
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
    (output_dir / "runtime_metadata.json").write_text(
        json.dumps(runtime_metadata, indent=2), encoding="utf-8"
    )

    all_rows: List[Dict[str, object]] = (
        load_completed_evaluation_rows(output_dir / "raw_metrics.csv", args.variants[0], config)
        if args.resume_evaluation
        else []
    )
    summaries: List[Dict[str, object]] = []
    resume_state: Optional[Dict[str, Any]] = None
    if args.resume_checkpoint is not None:
        resume_state = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
    training_rows: List[Dict[str, object]] = (
        list(resume_state["training_rows"]) if resume_state is not None else []
    )
    if training_rows:
        write_csv(output_dir / "training_metrics.csv", training_rows)

    def persist_training_row(row: Dict[str, object]) -> None:
        training_rows.append(row)
        write_csv(output_dir / "training_metrics.csv", training_rows)

    def persist_evaluation_row(row: Dict[str, object]) -> None:
        all_rows.append(row)
        write_csv(output_dir / "raw_metrics.csv", all_rows)

    for variant in args.variants:
        device = device_plan[variant]
        device_label = (
            f"cuda ({torch.cuda.get_device_name(device)})"
            if device.type == "cuda"
            else "cpu"
        )
        print(
            f"[{variant}] training ({config.learning.training_episodes} episodes) "
            f"on {device_label}"
        )
        checkpoint_path = output_dir / f"{variant}_seed{config.seed}_training.pth"
        model, _ = train_variant(
            variant,
            config,
            persist_training_row,
            checkpoint_path=checkpoint_path,
            checkpoint_every=args.checkpoint_every,
            resume_state=resume_state,
            device=device,
        )
        resume_state = None
        rows = (
            evaluate_variant(
                variant, model, config, persist_evaluation_row,
                start_episode=len(all_rows),
            )
            if config.learning.evaluation_episodes > 0
            else []
        )
        if rows or all_rows:
            summaries.append(summarize(all_rows, variant, config))
        atomic_torch_save(
            {
                "variant": variant,
                "config": config.to_dict(),
                "state_dict": model.state_dict(),
            },
            output_dir / f"{variant}_seed{config.seed}.pth",
        )
        # Persist each completed variant so an interruption never discards
        # already-finished pilot or formal evaluation data.
        if all_rows:
            write_csv(output_dir / "raw_metrics.csv", all_rows)
        if summaries:
            write_csv(output_dir / "run_summary.csv", summaries)
        print(f"[{variant}] evaluation complete")

    print(f"Results written to {output_dir.resolve()}")
    if config.mode != "full":
        print(
            f"{config.mode.upper()} ONLY: these values verify execution and must not "
            "be reported as paper results."
        )


if __name__ == "__main__":
    main()

