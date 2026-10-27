"""Paired, frozen-checkpoint formal evaluation of packet-level Proposed ablations."""

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .config import SimulationConfig
from .proposed_packet import PACKET_ABLATION_VARIANTS, ProposedPacketPolicy
from .run_packet_fair_formal import run as run_formal


PROJECT = Path(__file__).resolve().parent.parent
TRAINING_ROOT = PROJECT / "outputs" / "packet_ablation_load240_budget10m"
DEFAULT_SEEDS = (11, 22, 33, 44, 55)
DEFAULT_TRAFFIC_SEEDS = (101, 202, 303, 404, 505)


def load_ablation_policy(method: str, device: str, seed: int):
    if method not in PACKET_ABLATION_VARIANTS:
        raise ValueError(f"Not a packet-level ablation: {method}")
    folder = TRAINING_ROOT / f"seed{seed}" / method
    if not (folder / "training_complete.ok").is_file():
        raise RuntimeError(f"Training incomplete: {folder}")
    checkpoint = folder / "checkpoint_latest.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    signature = saved["signature"]
    simulation = SimulationConfig()
    if (signature["method"] != method or signature["seed"] != seed or
            signature["load_mbps"] != 240.0 or
            signature["warmup_slots"] != 30 or signature["train_slots"] != 100 or
            signature["max_train_decisions"] != 10_000_000 or
            signature["epsilon_decay_decisions"] != 8_000_000 or
            signature["simulation"] != asdict(simulation)):
        raise ValueError(f"Training signature mismatch: {checkpoint}")
    policy = ProposedPacketPolicy(
        simulation, seed, training=False, device=device,
        variant=PACKET_ABLATION_VARIANTS[method],
    )
    if saved["policy"]["name"] != method:
        raise ValueError(f"Checkpoint policy name mismatch: {checkpoint}")
    policy.model.load_state_dict(saved["policy"]["model_state"])
    policy.model.eval()
    if policy.training or policy.epsilon != 0.0:
        raise RuntimeError("Formal ablation evaluation must be greedy and inference-only")
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "train_decisions": int(saved["rows"][-1]["total_train_decisions"]),
        "train_episodes": int(saved["next_episode"]),
        "training_signature": signature,
    }
    del saved
    gc.collect()
    return policy, simulation, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-seeds", nargs="+", type=int,
                        default=list(DEFAULT_SEEDS))
    parser.add_argument("--traffic-seeds", nargs="+", type=int,
                        default=list(DEFAULT_TRAFFIC_SEEDS))
    parser.add_argument("--methods", nargs="+", choices=tuple(PACKET_ABLATION_VARIANTS),
                        default=list(PACKET_ABLATION_VARIANTS))
    parser.add_argument("--scenarios", nargs="+", choices=(
        "S1_non_cross_normal", "S2_cross_normal", "S3_non_cross_ftp", "S4_cross_ftp",
    ), default=["S1_non_cross_normal", "S2_cross_normal",
                "S3_non_cross_ftp", "S4_cross_ftp"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup-slots", type=int, default=30)
    parser.add_argument("--baseline-slots", type=int, default=250)
    parser.add_argument("--burst-slots", type=int, default=250)
    parser.add_argument("--recovery-slots", type=int, default=500)
    args = parser.parse_args()
    if not (TRAINING_ROOT / "training_complete.ok").is_file():
        raise RuntimeError("All 15 packet ablation trainings must finish before formal evaluation")
    try:
        run_formal(args, policy_loader=load_ablation_policy)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "FORMAL_FAILED.txt").write_text(repr(error) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
