"""Reproduce the paper's 60-second S4 packet-loss figure.

Loss is an eventual birth-cohort statistic.  For every plotted slot, the
numerator is eventual drops among packets born in the trailing ten measured
slots and the denominator is all births in those slots.  Packets still in
flight at the horizon are reported in the ledger but are not counted as loss.
The curve and band are the mean and sample SD across five paired runs.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np


SCENARIO = "S4_cross_ftp"
PAIRS = ((11, 101), (22, 202), (33, 303), (44, 404), (55, 505))
METHODS = (
    ("proposed_gat_lstm_packet", "Proposed GAT–LSTM–DQN", "#D32F2F", "-", "o"),
    ("graphpr_adapted", "GraphPR-adapted", "#1F77B4", "--", "s"),
    ("fdr_marl_adapted", "FDR-MARL-adapted", "#2E7D32", "-.", "^"),
    ("dqn_ir_fixed_link", "DQN-IR-adapted", "#7E57C2", ":", "D"),
)
EXPECTED = {
    "warmup_slots": 30,
    "baseline_slots": 500,
    "burst_slots": 250,
    "recovery_slots": 2250,
    "measurement_slots": 3000,
    "fixed_measurement_slots": 3000,
    "background_load_mbps": 240.0,
    "recovery_definition": "teacher_20261023",
}
SLOT_DURATION_S = 0.02
LOSS_WINDOW_SLOTS = 10


def configure() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 19,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 17,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rolling(values: list[float], width: int = LOSS_WINDOW_SLOTS) -> list[float]:
    result: list[float] = []
    for index in range(len(values)):
        window = values[max(0, index - width + 1): index + 1]
        result.append(sum(window) / width if len(window) == width else math.nan)
    return result


def derive_birth_cohorts(directory: Path, slots: int, warmup: int, cache: Path) -> list[dict[str, str]]:
    packaged = directory / "birth_cohorts.csv"
    if packaged.is_file():
        return read_csv(packaged)
    if cache.is_file():
        return read_csv(cache)
    packets = directory / "packets_FORMAL.csv.gz"
    if not packets.is_file():
        raise FileNotFoundError(f"Need birth_cohorts.csv or packets_FORMAL.csv.gz: {directory}")
    generated = [0] * slots
    dropped = [0] * slots
    pending = [0] * slots
    with gzip.open(packets, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        birth_index = header.index("birth_slot")
        outcome_index = header.index("outcome")
        for row in reader:
            index = int(row[birth_index]) - warmup
            if 0 <= index < slots:
                generated[index] += 1
                dropped[index] += row[outcome_index] == "dropped"
                pending[index] += row[outcome_index] == "in_flight_at_horizon"
    records = [
        {"index": index, "generated": generated[index], "dropped": dropped[index],
         "pending": pending[index]}
        for index in range(slots)
    ]
    write_csv(cache, records, ("index", "generated", "dropped", "pending"))
    return [{key: str(value) for key, value in record.items()} for record in records]


def validate_summary(summary: dict[str, object], method: str, checkpoint: int, traffic: int) -> None:
    identity = {
        "variant": method,
        "scenario": SCENARIO,
        "checkpoint_seed": checkpoint,
        "traffic_seed": traffic,
    }
    for key, value in identity.items():
        if summary.get(key) != value:
            raise ValueError(f"Unexpected {key} for {method}/{checkpoint}: {summary.get(key)!r}")
    for key, value in EXPECTED.items():
        if summary.get(key) != value:
            raise ValueError(f"Unexpected {key} for {method}/{checkpoint}: {summary.get(key)!r}")


def load_loss_series(root: Path, cache: Path) -> tuple[dict[str, list[list[float]]], int, int]:
    series = {method[0]: [] for method in METHODS}
    timings: set[tuple[int, int]] = set()
    for checkpoint, traffic in PAIRS:
        pair = root / f"checkpoint{checkpoint}_traffic{traffic}"
        manifest = json.loads((pair / "trace_manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("checkpoint_seed"), manifest.get("traffic_seed")) != (checkpoint, traffic):
            raise ValueError(f"Trace-manifest seed mismatch: {pair}")
        trace_hashes: set[str] = set()
        generated_by_method: list[list[float]] = []
        for method, *_ in METHODS:
            directory = pair / method / SCENARIO
            summary = json.loads((directory / "summary_FORMAL.json").read_text(encoding="utf-8"))
            validate_summary(summary, method, checkpoint, traffic)
            trace_hashes.add(str(summary["trace_sha256"]))
            rows = read_csv(directory / "slot_metrics_FORMAL.csv")
            if len(rows) != EXPECTED["measurement_slots"]:
                raise ValueError(f"Expected 3000 measured slots: {method}/{checkpoint}; got {len(rows)}")
            warmup = int(summary["warmup_slots"])
            measured_slots = [int(row["slot"]) for row in rows]
            if measured_slots != list(range(warmup, warmup + len(rows))):
                raise ValueError(f"Non-contiguous measured slots: {method}/{checkpoint}")
            start = int(summary["baseline_slots"])
            end = start + int(summary["burst_slots"])
            timings.add((start, end))
            cohort = derive_birth_cohorts(
                directory, len(rows), warmup,
                cache / f"{SCENARIO}_{method}_{checkpoint}.csv",
            )
            if len(cohort) != len(rows):
                raise ValueError(f"Birth-cohort length mismatch: {method}/{checkpoint}")
            generated = [float(row["generated_packets"]) for row in rows]
            cohort_generated = [float(row["generated"]) for row in cohort]
            if generated != cohort_generated:
                raise ValueError(f"Birth-cohort ledger mismatch: {method}/{checkpoint}")
            if any(int(row["dropped"]) + int(row["pending"]) > int(row["generated"])
                   for row in cohort):
                raise ValueError(f"Invalid cohort outcomes: {method}/{checkpoint}")
            drops = [float(row["dropped"]) for row in cohort]
            totals = rolling(generated)
            loss = [100.0 * dropped / total if total > 0 else math.nan
                    for dropped, total in zip(rolling(drops), totals)]
            series[method].append(loss)
            generated_by_method.append(generated)
        if trace_hashes != {str(manifest["trace_sha256"])}:
            raise ValueError(f"Methods do not use the paired trace: {checkpoint}/{traffic}")
        if any(values != generated_by_method[0] for values in generated_by_method[1:]):
            raise ValueError(f"Generated traffic is not paired across methods: {checkpoint}/{traffic}")
    if len(timings) != 1:
        raise ValueError(f"Burst timings differ: {timings}")
    start, end = next(iter(timings))
    return series, start, end


def blend_with_white(color: str, alpha: float) -> tuple[float, float, float]:
    return tuple(alpha * channel + (1.0 - alpha) for channel in to_rgb(color))


def aggregate(series: dict[str, list[list[float]]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method, *_ in METHODS:
        values = np.asarray(series[method], dtype=float)
        mean = np.full(values.shape[1], np.nan)
        sd = np.full(values.shape[1], np.nan)
        mean[LOSS_WINDOW_SLOTS - 1:] = np.mean(values[:, LOSS_WINDOW_SLOTS - 1:], axis=0)
        sd[LOSS_WINDOW_SLOTS - 1:] = np.std(values[:, LOSS_WINDOW_SLOTS - 1:], axis=0, ddof=1)
        result[method] = mean, sd
    return result


def build(statistics: dict[str, tuple[np.ndarray, np.ndarray]], start: int, end: int,
          *, eps: bool) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9.2, 6.2))
    length = min(len(mean) for mean, _ in statistics.values())
    time = np.arange(length) * SLOT_DURATION_S
    for method, label, color, linestyle, marker in METHODS:
        mean, sd = statistics[method]
        axis.plot(time[::5], mean[::5], label=label, color=color, linestyle=linestyle,
                  marker=marker, markevery=10, linewidth=2, markersize=5.5,
                  markerfacecolor="white", markeredgewidth=1.2)
        band_color = blend_with_white(color, 0.12) if eps else color
        axis.fill_between(time, mean - sd, mean + sd, color=band_color,
                          alpha=1.0 if eps else 0.12, linewidth=0)
    span_color = blend_with_white("#F2F2F2", 0.35) if eps else "#F2F2F2"
    axis.axvspan(start * SLOT_DURATION_S, end * SLOT_DURATION_S,
                 color=span_color, alpha=1.0 if eps else 0.35)
    for x in (start * SLOT_DURATION_S, end * SLOT_DURATION_S):
        axis.axvline(x, color="#666666", linestyle="--", linewidth=1.2)
    axis.set_xlim(0, 60)
    axis.set_xticks(np.arange(0, 61, 10))
    axis.set_xlabel("Time since measurement start (s)")
    axis.set_ylabel("Packet loss rate (%)")
    axis.set_ylim(bottom=0)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                columnspacing=1.4, handlelength=2.0)
    figure.subplots_adjust(left=0.13, right=0.99, bottom=0.34, top=0.97)
    return figure


def process(root: Path, output: Path) -> dict[str, object]:
    root = Path(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configure()
    series, start, end = load_loss_series(root, output / "cohort_cache")
    statistics = aggregate(series)

    csv_rows: list[dict[str, object]] = []
    for index in range(EXPECTED["measurement_slots"]):
        row: dict[str, object] = {"slot_index": index, "time_s": index * SLOT_DURATION_S}
        for method, *_ in METHODS:
            mean, sd = statistics[method]
            row[f"{method}_mean_percent"] = mean[index]
            row[f"{method}_sample_sd_percent"] = sd[index]
        csv_rows.append(row)
    fields = ("slot_index", "time_s", *(
        field for method, *_ in METHODS
        for field in (f"{method}_mean_percent", f"{method}_sample_sd_percent")
    ))
    write_csv(output / "S4_cross_ftp_loss_mean_sd.csv", csv_rows, fields)

    base = output / "BURST_S4_cross_ftp_loss"
    figure = build(statistics, start, end, eps=False)
    figure.savefig(base.with_suffix(".pdf"))
    figure.savefig(base.with_suffix(".png"), dpi=180)
    plt.close(figure)
    figure = build(statistics, start, end, eps=True)
    figure.savefig(base.with_suffix(".eps"))
    plt.close(figure)

    start_15 = round(15.0 / SLOT_DURATION_S)
    interval_means = {
        method: float(np.mean(statistics[method][0][start_15:]))
        for method, *_ in METHODS
    }
    report: dict[str, object] = {
        "scenario": SCENARIO,
        "duration_s": 60,
        "measured_slots": EXPECTED["measurement_slots"],
        "warmup_slots_excluded": EXPECTED["warmup_slots"],
        "slot_duration_s": SLOT_DURATION_S,
        "seed_pairs": [{"checkpoint_seed": checkpoint, "traffic_seed": traffic}
                       for checkpoint, traffic in PAIRS],
        "loss_definition": (
            "eventual dropped births / all births in each trailing 10-slot (0.2-s) "
            "birth-cohort window; in-flight-at-horizon packets are censored, not losses"
        ),
        "aggregation": "five paired runs; curve = mean; band = sample SD (ddof=1)",
        "burst_interval_s": [start * SLOT_DURATION_S, end * SLOT_DURATION_S],
        "mean_loss_15_to_60_percent": interval_means,
        "proposed_below_graphpr_15_to_60": (
            interval_means["proposed_gat_lstm_packet"] < interval_means["graphpr_adapted"]
        ),
    }
    (output / "fig4_60s_verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    process(arguments.root, arguments.output)


if __name__ == "__main__":
    main()
