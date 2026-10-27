"""Create dependency-free static SVG time curves after formal evaluation."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict, deque
from html import escape
from pathlib import Path

from .robustness_aggregate import SEED_PAIRS, VARIANTS
from .robustness_pilot import SCENARIOS


SCENARIO_STYLE = {
    "S1_non_cross_normal": ("S1 same · normal", "#1f5d91", ""),
    "S2_cross_normal": ("S2 cross · normal", "#c05a26", ""),
    "S3_non_cross_ftp": ("S3 same · FTP", "#1f5d91", "12 7"),
    "S4_cross_ftp": ("S4 cross · FTP", "#c05a26", "12 7"),
}
METRICS = {
    "queue": ("Average queue length (packets/satellite)", "queue_20_slot_mean"),
    "loss": ("Cumulative observed packet drops (%)", "cumulative_drop_fraction"),
    "delay": ("End-to-end delay (ms; delivered, 20-slot mean)", "delay_20_slot_mean"),
}


def smoothed(rows):
    generated = dropped = 0
    queue_window = deque(maxlen=20)
    delivered_window = deque(maxlen=20)
    delay_sum_window = deque(maxlen=20)
    result = {}
    burst_start = next(int(row["slot"]) for row in rows if row["phase"] == "burst")
    for row in rows:
        slot = int(row["slot"])
        generated += int(row["generated_packets"])
        dropped += int(row["dropped_packets"])
        queue_window.append(float(row["average_queue_packets"]))
        delivered = int(row["delivered_packets"])
        delivered_window.append(delivered)
        delay_sum_window.append(
            float(row["mean_delivered_delay_ms"]) * delivered
            if row["mean_delivered_delay_ms"] else 0.0
        )
        recent_delivered = sum(delivered_window)
        result[slot - burst_start] = {
            "queue_20_slot_mean": statistics.mean(queue_window),
            "cumulative_drop_fraction": 100.0 * dropped / generated if generated else 0.0,
            "delay_20_slot_mean": (
                sum(delay_sum_window) / recent_delivered if recent_delivered else math.nan
            ),
        }
    return result


def aggregate_series(root, variant, scenario):
    per_seed = []
    for checkpoint_seed, traffic_seed in SEED_PAIRS:
        path = (root / f"checkpoint{checkpoint_seed}_traffic{traffic_seed}"
                / variant / scenario / "slot_metrics_FORMAL.csv")
        with path.open(newline="", encoding="utf-8-sig") as stream:
            per_seed.append(smoothed(list(csv.DictReader(stream))))
    common_keys = sorted(set.intersection(*(set(series) for series in per_seed)))
    result = {metric: [] for metric in METRICS}
    for key in common_keys:
        for metric, (_, field) in METRICS.items():
            values = [series[key][field] for series in per_seed]
            values = [value for value in values if math.isfinite(value)]
            result[metric].append((key * 0.02, statistics.mean(values) if values else math.nan))
    return result


def svg_chart(series_by_scenario, metric):
    width, height = 1200, 680
    left, right, top, bottom = 125, 1160, 85, 575
    values = [value for series in series_by_scenario.values() for _, value in series[metric]
              if math.isfinite(value)]
    times = [time for series in series_by_scenario.values() for time, _ in series[metric]]
    if not values or not times:
        raise RuntimeError("No finite chart data")
    xmin, xmax = min(times), max(times)
    ymin, ymax = 0.0, max(values) * 1.08
    if ymax <= 0:
        ymax = 1.0

    def xmap(value):
        return left + (value - xmin) / (xmax - xmin) * (right - left)

    def ymap(value):
        return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(METRICS[metric][0])} over time for four scenarios">',
        '<rect width="1200" height="680" fill="white"/>',
        f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" '
        'fill="none" stroke="#333" stroke-width="1.3"/>',
    ]
    for index in range(6):
        value = ymin + (ymax - ymin) * index / 5
        y = ymap(value)
        pieces.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" '
                      'stroke="#d9d9d9" stroke-width="1"/>')
        pieces.append(f'<text x="{left-12}" y="{y+6:.2f}" text-anchor="end" '
                      f'font-family="Times New Roman" font-size="17">{value:.1f}</text>')
    for tick in (-5, 0, 5, 10, 15):
        if xmin <= tick <= xmax:
            x = xmap(tick)
            pieces.append(f'<line x1="{x:.2f}" y1="{bottom}" x2="{x:.2f}" y2="{bottom+7}" '
                          'stroke="#333" stroke-width="1"/>')
            pieces.append(f'<text x="{x:.2f}" y="{bottom+29}" text-anchor="middle" '
                          f'font-family="Times New Roman" font-size="17">{tick}</text>')
    for scenario, _, _ in SCENARIOS:
        label, color, dash = SCENARIO_STYLE[scenario]
        points = " ".join(
            f"{xmap(time):.2f},{ymap(value):.2f}"
            for time, value in series_by_scenario[scenario][metric]
            if math.isfinite(value)
        )
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        pieces.append(f'<polyline points="{points}" fill="none" stroke="{color}" '
                      f'stroke-width="2.6"{dash_attr}/>')
    legend_x = (165, 425, 685, 945)
    for x, (scenario, _, _) in zip(legend_x, SCENARIOS):
        label, color, dash = SCENARIO_STYLE[scenario]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        pieces.append(f'<line x1="{x}" y1="42" x2="{x+42}" y2="42" '
                      f'stroke="{color}" stroke-width="3"{dash_attr}/>')
        pieces.append(f'<text x="{x+50}" y="48" font-family="Times New Roman" '
                      f'font-size="17">{escape(label)}</text>')
    pieces.append(f'<text x="{(left+right)/2:.2f}" y="{height-30}" text-anchor="middle" '
                  'font-family="Times New Roman" font-size="20">Time from burst start (s)</text>')
    pieces.append(f'<text x="32" y="{(top+bottom)/2:.2f}" '
                  f'transform="rotate(-90 32 {(top+bottom)/2:.2f})" text-anchor="middle" '
                  f'font-family="Times New Roman" font-size="20">'
                  f'{escape(METRICS[metric][0])}</text>')
    pieces.append('</svg>')
    return "\n".join(pieces) + "\n"


def svg_recovery(root):
    grouped = defaultdict(list)
    for checkpoint_seed, traffic_seed in SEED_PAIRS:
        for variant in VARIANTS:
            for scenario in ("S3_non_cross_ftp", "S4_cross_ftp"):
                path = (root / f"checkpoint{checkpoint_seed}_traffic{traffic_seed}"
                        / variant / scenario / "summary_FORMAL.json")
                record = json.loads(path.read_text(encoding="utf-8"))
                if record["recovery_status"] == "recovered":
                    grouped[(variant, scenario)].append(float(record["recovery_time_s"]))
    width, height = 1200, 680
    left, right, top, bottom = 125, 1160, 85, 575
    ymax = max(max(values) for values in grouped.values()) * 1.1
    ymax = math.ceil(ymax)

    def ymap(value):
        return bottom - value / ymax * (bottom - top)

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Recovery time by model and region">',
        '<rect width="1200" height="680" fill="white"/>',
        f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" '
        'fill="none" stroke="#333" stroke-width="1.3"/>',
    ]
    for index in range(ymax + 1):
        y = ymap(index)
        pieces.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" '
                      'stroke="#e0e0e0" stroke-width="1"/>')
        pieces.append(f'<text x="{left-12}" y="{y+6:.2f}" text-anchor="end" '
                      f'font-family="Times New Roman" font-size="17">{index}</text>')
    x_centers = (250, 500, 750, 1000)
    labels = ("DQN", "LSTM-DQN", "GAT-DQN", "GAT-LSTM-DQN")
    for variant, center, label in zip(VARIANTS, x_centers, labels):
        pieces.append(f'<text x="{center}" y="{bottom+32}" text-anchor="middle" '
                      f'font-family="Times New Roman" font-size="18">{label}</text>')
        for scenario, offset, color in (
            ("S3_non_cross_ftp", -35, "#1f5d91"),
            ("S4_cross_ftp", 35, "#c05a26"),
        ):
            values = grouped[(variant, scenario)]
            x = center + offset
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            pieces.append(f'<line x1="{x}" y1="{ymap(mean+sd):.2f}" '
                          f'x2="{x}" y2="{ymap(max(0,mean-sd)):.2f}" '
                          f'stroke="{color}" stroke-width="2.4"/>')
            for value in (mean+sd, max(0,mean-sd)):
                pieces.append(f'<line x1="{x-11}" y1="{ymap(value):.2f}" '
                              f'x2="{x+11}" y2="{ymap(value):.2f}" '
                              f'stroke="{color}" stroke-width="2.4"/>')
            pieces.append(f'<line x1="{x-14}" y1="{ymap(mean):.2f}" '
                          f'x2="{x+14}" y2="{ymap(mean):.2f}" '
                          f'stroke="{color}" stroke-width="3.5"/>')
            for index, value in enumerate(values):
                dot_x = x + (index - (len(values)-1)/2)*5
                if scenario == "S3_non_cross_ftp":
                    pieces.append(f'<circle cx="{dot_x:.2f}" cy="{ymap(value):.2f}" '
                                  f'r="4.5" fill="{color}"/>')
                else:
                    y = ymap(value)
                    points = f"{dot_x:.2f},{y-5:.2f} {dot_x-5:.2f},{y+5:.2f} {dot_x+5:.2f},{y+5:.2f}"
                    pieces.append(f'<polygon points="{points}" fill="{color}"/>')
    pieces += [
        '<circle cx="170" cy="42" r="5" fill="#1f5d91"/>',
        '<text x="185" y="48" font-family="Times New Roman" font-size="17">Same-region FTP</text>',
        '<polygon points="453,37 448,47 458,47" fill="#c05a26"/>',
        '<text x="468" y="48" font-family="Times New Roman" font-size="17">Cross-region FTP</text>',
        f'<text x="{(left+right)/2:.2f}" y="{height-30}" text-anchor="middle" '
        'font-family="Times New Roman" font-size="20">Model</text>',
        f'<text x="32" y="{(top+bottom)/2:.2f}" '
        f'transform="rotate(-90 32 {(top+bottom)/2:.2f})" text-anchor="middle" '
        'font-family="Times New Roman" font-size="20">Recovery time (s)</text>',
        '</svg>',
    ]
    return "\n".join(pieces) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if not (root / "FORMAL_COMPLETE.ok").exists():
        raise RuntimeError("Formal evaluation is incomplete")
    figures = root / "time_curves"
    figures.mkdir(exist_ok=True)
    for variant in VARIANTS:
        series = {scenario: aggregate_series(root, variant, scenario)
                  for scenario, _, _ in SCENARIOS}
        for metric in METRICS:
            (figures / f"{variant}_{metric}_time.svg").write_text(
                svg_chart(series, metric), encoding="utf-8"
            )
    (figures / "recovery_time.svg").write_text(svg_recovery(root), encoding="utf-8")
    print(f"Wrote {len(VARIANTS)*len(METRICS)} time-curve SVGs and recovery plot in {figures}")


if __name__ == "__main__":
    main()
