"""Paired, frozen-checkpoint formal evaluation of four packet-routing methods."""

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .packet_validation import METHODS, load_frozen_policy
from .robustness_formal import run_scenario, trace_digest
from .robustness_pilot import PilotSettings, SCENARIOS, make_trace, write_csv


DEFAULT_CHECKPOINT_SEEDS = (11, 22, 33, 44, 55)
DEFAULT_TRAFFIC_SEEDS = (101, 202, 303, 404, 505)


def run(args, *, policy_loader=None):
    if policy_loader is None:
        policy_loader = load_frozen_policy
    if (len(args.checkpoint_seeds) != len(args.traffic_seeds) or
            len(set(args.checkpoint_seeds)) != len(args.checkpoint_seeds) or
            len(set(args.traffic_seeds)) != len(args.traffic_seeds)):
        raise ValueError("Checkpoint and traffic seeds must be unique paired lists")
    if len(set(args.methods)) != len(args.methods) or len(set(args.scenarios)) != len(args.scenarios):
        raise ValueError("Methods and scenarios must not repeat")
    if min(args.warmup_slots, args.baseline_slots, args.burst_slots,
           args.recovery_slots) <= 0:
        raise ValueError("All durations must be positive")
    if getattr(args, "measurement_slots", None) is not None and args.measurement_slots <= 0:
        raise ValueError("Fixed measurement window must be positive")
    if getattr(args, "background_load_mbps", 240.0) < 0:
        raise ValueError("Background load cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = {"torch_version": torch.__version__, "device": args.device,
               "methods": list(args.methods), "scenarios": list(args.scenarios),
               "measurement_slots": getattr(args, "measurement_slots", None),
               "recovery_definition": getattr(args, "recovery_definition", "legacy")}
    runtime_path = output / "evaluation_runtime.json"
    if runtime_path.exists() and json.loads(runtime_path.read_text(encoding="utf-8")) != runtime:
        raise RuntimeError("Evaluation runtime/protocol differs from the existing batch")
    runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    (output / "FORMAL_FAILED.txt").unlink(missing_ok=True)
    scenarios = [row for row in SCENARIOS if row[0] in args.scenarios]
    if len(scenarios) != len(args.scenarios):
        raise ValueError("Unknown scenario")
    results = []
    total = len(args.checkpoint_seeds) * len(args.methods) * len(scenarios)
    for checkpoint_seed, traffic_seed in zip(args.checkpoint_seeds, args.traffic_seeds):
        settings = PilotSettings(
            seed=traffic_seed, background_load_mbps=getattr(args, "background_load_mbps", 240.0),
            warmup_slots=args.warmup_slots, baseline_slots=args.baseline_slots,
            burst_slots=args.burst_slots, recovery_slots=args.recovery_slots,
            measurement_slots=getattr(args, "measurement_slots", None),
            recovery_definition=getattr(args, "recovery_definition", "legacy"),
        )
        for method in args.methods:
            policy, simulation, metadata = policy_loader(method, args.device,
                                                         checkpoint_seed)
            trace = make_trace(settings, simulation)
            digest = trace_digest(trace)
            pair_dir = output / f"checkpoint{checkpoint_seed}_traffic{traffic_seed}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            manifest = pair_dir / "trace_manifest.json"
            manifest_data = {
                "checkpoint_seed": checkpoint_seed, "traffic_seed": traffic_seed,
                "settings": asdict(settings), "trace_sha256": digest,
                "burst_start": trace.burst_start, "burst_end": trace.burst_end,
                "recovery_start": trace.recovery_start, "horizon": trace.horizon,
                "same_region_destinations": list(trace.same_region_destinations),
                "cross_region_destinations": list(trace.cross_region_destinations),
                "ftp_starts": {str(k): list(v) for k, v in trace.ftp_starts.items()},
                "nhpp_counts": trace.nhpp_counts,
            }
            manifest_data = json.loads(json.dumps(manifest_data, ensure_ascii=False))
            if manifest.exists():
                previous = json.loads(manifest.read_text(encoding="utf-8"))
                if previous != manifest_data:
                    raise RuntimeError(f"Incompatible paired trace: {manifest}")
            else:
                manifest.write_text(json.dumps(manifest_data, ensure_ascii=False),
                                    encoding="utf-8")
            for name, cross, use_ftp in scenarios:
                (output / "progress.json").write_text(json.dumps({
                    "completed": len(results), "total": total, "status": "running",
                    "current_checkpoint_seed": checkpoint_seed,
                    "current_traffic_seed": traffic_seed, "current_method": method,
                    "current_scenario": name,
                }, indent=2), encoding="utf-8")
                print(f"[{len(results)+1}/{total}] {checkpoint_seed}/{traffic_seed}/"
                      f"{method}/{name} starting", flush=True)
                result = run_scenario(
                    name, cross, use_ftp, method, policy, checkpoint_seed, settings,
                    simulation, trace, digest, output,
                    checkpoint_override=Path(metadata["checkpoint"]),
                )
                for key in ("throughput_mbps", "packet_loss_rate",
                            "end_to_end_delay_ms", "average_queue_packets"):
                    if result[key] is None:
                        raise RuntimeError(f"Undefined {key}: {method}/{name}")
                results.append(result)
                (output / "progress.json").write_text(json.dumps({
                    "completed": len(results), "total": total,
                    "last_checkpoint_seed": checkpoint_seed,
                    "last_traffic_seed": traffic_seed,
                    "last_method": method, "last_scenario": name,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[{len(results)}/{total}] complete", flush=True)
            del policy
            gc.collect()
    for checkpoint_seed, traffic_seed in zip(args.checkpoint_seeds, args.traffic_seeds):
        pair = [row for row in results if row["checkpoint_seed"] == checkpoint_seed]
        for name, _, _ in scenarios:
            matches = [row for row in pair if row["scenario"] == name]
            if len({row["generated_packets"] for row in matches}) != 1:
                raise RuntimeError(f"Traffic not paired: {checkpoint_seed}/{name}")
            if len({row["trace_sha256"] for row in matches}) != 1:
                raise RuntimeError(f"Trace not paired: {checkpoint_seed}/{name}")
    write_csv(output / "summary_FORMAL.csv", [
        {key: value for key, value in row.items()
         if key not in ("birth_phase_counts", "traffic_class_counts")}
        for row in results
    ])
    (output / "FORMAL_COMPLETE.ok").write_text(
        f"All {total} paired frozen-checkpoint scenarios completed.\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-seeds", nargs="+", type=int,
                        default=list(DEFAULT_CHECKPOINT_SEEDS))
    parser.add_argument("--traffic-seeds", nargs="+", type=int,
                        default=list(DEFAULT_TRAFFIC_SEEDS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--scenarios", nargs="+", choices=[row[0] for row in SCENARIOS],
                        default=[row[0] for row in SCENARIOS])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup-slots", type=int, default=30)
    parser.add_argument("--baseline-slots", type=int, default=250)
    parser.add_argument("--burst-slots", type=int, default=250)
    parser.add_argument("--recovery-slots", type=int, default=500)
    parser.add_argument("--background-load-mbps", type=float, default=240.0)
    parser.add_argument("--measurement-slots", type=int)
    parser.add_argument("--recovery-definition", choices=("legacy", "teacher_20261023"), default="legacy")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "FORMAL_FAILED.txt").write_text(repr(error) + "\n",
                                                           encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
