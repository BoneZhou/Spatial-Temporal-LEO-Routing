"""Small, portable entry point for the paper's packet-level experiments."""
import argparse
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def invoke(module, arguments, dry_run=False):
    command = [sys.executable, "-X", "utf8", "-m", "icc2027_experiments." + module]
    command += [str(value) for value in arguments]
    if dry_run:
        print(json.dumps(command))
    else:
        subprocess.run(command, cwd=ROOT, check=True)


def options(settings):
    result = []
    for key, value in settings.items():
        result.append("--" + key.replace("_", "-"))
        result.extend(value if isinstance(value, list) else [value])
    return result


def training_directory(config, seed, method):
    paths = config["paths"]
    if method in config["ablations"]:
        return ROOT / paths["ablation_training"] / f"seed{seed}" / method
    base = (ROOT / paths["seed11_training"] if seed == 11 else
            ROOT / paths["multiseed_training"] / f"seed{seed}")
    return base / method


def train(args, config):
    methods = config["methods"] if args.group == "main" else config["ablations"]
    for seed in args.seeds or config["training_seeds"]:
        if seed not in config["training_seeds"]:
            raise ValueError("Use one of the configured training seeds")
        for method in methods:
            output = training_directory(config, seed, method)
            if (output / "training_complete.ok").is_file():
                print(f"Already complete: {seed}/{method}")
                continue
            command = ["--method", method, "--seed", seed, "--device", args.device,
                       "--output-dir", output, *options(config["training"])]
            if (output / "checkpoint_latest.pt").is_file():
                command.append("--resume")
            invoke("baseline_train", command, args.dry_run)
    if args.group == "ablation" and not args.dry_run:
        complete = all((training_directory(config, seed, method) /
                        "training_complete.ok").is_file()
                       for seed in config["training_seeds"] for method in methods)
        if complete:
            (ROOT / config["paths"]["ablation_training"] /
             "training_complete.ok").write_text("All configured ablations complete.\n")


def audit(args, config, name):
    methods = config["ablations"] if name == "ablation" else config["methods"]
    scenarios = config["burst" if name == "burst" else "main"]["scenarios"]
    invoke("robustness_audit", [
        "--output-dir", config["paths"][name], "--variants", *methods,
        "--scenarios", *scenarios, "--checkpoint-seeds", *config["training_seeds"],
        "--traffic-seeds", *config["traffic_seeds"]], args.dry_run)


def evaluate(args, config, name):
    # Existing loaders deliberately check the paper's frozen training signature.
    from icc2027_experiments import packet_validation as main_loader
    from icc2027_experiments import run_packet_ablation_formal as ablation_loader
    expected = {
        "seed11_training": main_loader.TRAINING_ROOT,
        "multiseed_training": ROOT / "outputs/packet_fair_multiseed_load240_budget10m",
        "ablation_training": ablation_loader.TRAINING_ROOT,
    }
    if any((ROOT / config["paths"][key]).resolve() != path.resolve()
           for key, path in expected.items()):
        raise ValueError("Training locations must match frozen-policy loader conventions")
    if config["training"]["load_mbps"] != 240:
        raise ValueError("Paper evaluation requires 240-Mbps trained checkpoints")
    settings = dict(config["burst" if name == "burst" else "main"])
    methods = config["ablations"] if name == "ablation" else config["methods"]
    module = ("run_packet_ablation_formal" if name == "ablation" else
              "run_packet_fair_formal")
    # The ablation CLI implements the original aggregate timing.
    if name == "ablation":
        settings = {key: value for key, value in settings.items()
                    if key not in ("background_load_mbps", "recovery_definition")}
    invoke(module, [
        "--output-dir", config["paths"][name], "--device", args.device,
        "--checkpoint-seeds", *config["training_seeds"],
        "--traffic-seeds", *config["traffic_seeds"], "--methods", *methods,
        *options(settings)], args.dry_run)
    audit(args, config, name)


def tables(args, config):
    from icc2027_experiments.packet_formal_aggregate import aggregate
    from icc2027_experiments.packet_formal_ci import analyze, write_outputs
    from icc2027_experiments.packet_ablation_aggregate import aggregate as ablate
    from icc2027_experiments.packet_cross_region_paired import paired_analysis
    from icc2027_experiments.paired_cross_region_ci import analyze as cross_ci
    from icc2027_experiments.paired_cross_region_ci import write_outputs as write_cross_ci
    import shutil
    if args.source == "reference":
        root = ROOT / "outputs/reference"
        # Copy the compact snapshots so analysis never edits the reference data.
        for name in ("main", "ablation"):
            shutil.copytree(ROOT / "data" / name, root / name, dirs_exist_ok=True)
        main, ablation = root / "main", root / "ablation"
    else:
        main = ROOT / config["paths"]["main"]
        ablation = ROOT / config["paths"]["ablation"]
    aggregate(main)
    write_outputs(main / "ci95", analyze(main))
    paired = paired_analysis(main)
    write_cross_ci(cross_ci(paired[0], paired[1]), main / "cross_region_ci95")
    ablate(ROOT, ablation, full_root=main, new_root=ablation)
    print(f"Tables and CSV: {main} ; {ablation}")


def plot(args, config):
    from icc2027_experiments.publication_style import MAIN, ABLATION, configure
    from icc2027_experiments.revision_figures import draw
    from icc2027_experiments.paper_plots import combined, training_curves
    configure()
    if args.source == "reference":
        main, ablation = ROOT / "outputs/reference/main", ROOT / "outputs/reference/ablation"
    else:
        main, ablation = (ROOT / config["paths"][name] for name in ("main", "ablation"))
    for name, source, filename, methods in (
        ("main", main, "four_metrics_mean_sd.csv", MAIN),
        ("ablation", ablation, "ablation_four_metrics_mean_sd.csv", ABLATION),
    ):
        output = ROOT / "outputs/figures" / args.source / name
        draw(source / filename, output, methods)
        combined(source / filename, output, methods)
    if args.source == "reference":
        training_curves(ROOT / "data/training", ROOT / "outputs/figures/reference/training")
    print("Figures: outputs/figures/" + args.source)


def burst_plot(args, config):
    from icc2027_experiments.fig4_60s import process
    source = (ROOT / "data/fig4_60s" if args.source == "reference"
              else ROOT / config["paths"]["burst"])
    process(source, ROOT / "outputs/figures" / args.source / "fig4_60s")


def burst_recovery_plot(args, config):
    """Retain the earlier 30-second S3/S4 recovery diagnostic."""
    from icc2027_experiments.publication_style import configure
    from icc2027_experiments.revision_burst_analysis import process
    configure()
    source = (ROOT / "data/burst" if args.source == "reference"
              else ROOT / config["paths"]["burst_recovery_30s"])
    process(source, ROOT / "outputs/figures" / args.source / "burst_recovery_30s")


def check(config):
    from icc2027_experiments.config import BASELINE_LEARNING, LearningConfig, SimulationConfig
    from icc2027_experiments.robustness_pilot import PilotSettings
    for key, value in config["ftp"].items():
        if getattr(PilotSettings(), key) != value:
            raise ValueError(f"FTP setting disagrees with evaluator: {key}")
    if len(config["training_seeds"]) != len(config["traffic_seeds"]):
        raise ValueError("Training and traffic seeds must be paired")
    expected_burst = {
        "warmup_slots": 30, "baseline_slots": 500, "burst_slots": 250,
        "recovery_slots": 2250, "measurement_slots": 3000,
        "background_load_mbps": 240, "recovery_definition": "teacher_20261023",
        "scenarios": ["S4_cross_ftp"],
    }
    if config["burst"] != expected_burst:
        raise ValueError("The paper Fig. 4 protocol must remain the documented 60-second S4 run")
    expected_analysis = {
        "slot_duration_s": 0.02, "loss_window_slots": 10, "loss_window_s": 0.2,
        "band": "mean_plus_minus_sample_sd", "sample_sd_ddof": 1,
        "in_flight_treatment": "right_censored_not_loss",
        "recovery_baseline_slots": 50, "recovery_queue_window_slots": 25,
        "recovery_stable_slots": 25,
    }
    if config["fig4_analysis"] != expected_analysis:
        raise ValueError("Fig. 4 analysis settings disagree with the released plotting code")
    print(json.dumps({
        "simulation": asdict(SimulationConfig()),
        "proposed_learning": asdict(LearningConfig()),
        "adapted_learning": BASELINE_LEARNING,
        "paper_protocol": config,
    }, indent=2))
    for module in ("baseline_train", "run_packet_fair_formal", "run_packet_ablation_formal",
                   "robustness_formal", "robustness_audit", "revision_burst_analysis",
                   "fig4_60s"):
        importlib.import_module("icc2027_experiments." + module)


def validate_config(config):
    from icc2027_experiments.robustness_pilot import PilotSettings
    for key, value in config["ftp"].items():
        if getattr(PilotSettings(), key) != value:
            raise ValueError(f"FTP setting differs from the trace generator: {key}")
    pairs = list(zip(config["training_seeds"], config["traffic_seeds"]))
    if pairs != [(11, 101), (22, 202), (33, 303), (44, 404), (55, 505)]:
        raise ValueError("Paper statistics require the five documented seed pairs")
    for value in config["paths"].values():
        path = Path(value)
        if path.is_absolute() or not (ROOT / path).resolve().is_relative_to(ROOT):
            raise ValueError("Outputs must remain inside the repository")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=(
        "check", "smoke", "train", "main", "ablation", "burst",
        "tables", "plot", "burst-plot", "burst-recovery-plot", "test"))
    parser.add_argument("--config", type=Path, default=ROOT / "paper_config.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--group", choices=("main", "ablation"), default="main")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--source", choices=("reference", "fresh"), default="reference")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    os.chdir(ROOT)
    validate_config(config)
    if args.action == "train":
        train(args, config)
    elif args.action in ("main", "ablation", "burst"):
        evaluate(args, config, args.action)
    elif args.action == "tables":
        tables(args, config)
    elif args.action == "plot":
        plot(args, config)
    elif args.action == "burst-plot":
        burst_plot(args, config)
    elif args.action == "burst-recovery-plot":
        burst_recovery_plot(args, config)
    elif args.action == "check":
        check(config)
    elif args.action == "test":
        subprocess.run([sys.executable, "-m", "unittest", "discover",
                        "-s", "icc2027_experiments", "-t", "."], check=True)
    elif args.action == "smoke":
        for method in config["methods"] + config["ablations"]:
            invoke("baseline_train", [
                "--method", method, "--seed", 11, "--load-mbps", 20,
                "--episodes", 1, "--warmup-slots", 1, "--train-slots", 2,
                "--batch-size", 2, "--update-every-slots", 1, "--device", args.device,
                "--output-dir", ROOT / "outputs/smoke" / method], args.dry_run)


if __name__ == "__main__":
    main()
