"""Draw paper-ready paired cross-region figures from the audited summary CSV."""

import argparse
import csv
import math
from html import escape
from pathlib import Path


METHODS = (
    ("proposed_gat_lstm_packet", "Proposed", "#c92f37"),
    ("graphpr_adapted", "GraphPR-adapted", "#2374aa"),
    ("fdr_marl_adapted", "FDR-MARL-adapted", "#42944a"),
    ("dqn_ir_fixed_link", "DQN-IR-adapted (fixed-link)", "#8a58a7"),
)
TRAFFIC = ("normal", "ftp")
METRICS = (
    ("throughput_mbps", "Throughput change (Mbps)", "throughput_difference"),
    ("packet_loss_rate", "Loss change (percentage points)",
     "packet_loss_difference"),
    ("end_to_end_delay_ms", "End-to-end delay change (ms)",
     "end_to_end_delay_difference"),
    ("average_queue_packets", "Queue change (packets/sat.)",
     "mean_queue_length_difference"),
)


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(method, traffic) for method, _, _ in METHODS for traffic in TRAFFIC}
    actual = {(row["method"], row["traffic"]) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("Expected one normal and one FTP paired row for each method")
    for row in rows:
        if int(row["n_pairs"]) != 5:
            raise ValueError("Each paired summary must contain five seeds")
        for metric, _, _ in METRICS:
            for suffix in ("mean", "sd"):
                value = float(row[f"{metric}_paired_delta_{suffix}"])
                if not math.isfinite(value) or (suffix == "sd" and value < 0):
                    raise ValueError(f"Invalid {metric} {suffix}")
    return {(row["method"], row["traffic"]): row for row in rows}


def nice_axis(values):
    low = min(0.0, *(mean - sd for mean, sd in values))
    high = max(0.0, *(mean + sd for mean, sd in values))
    span = high - low
    if span == 0:
        return -1.0, 1.0, 0.5
    rough = span / 5
    magnitude = 10 ** math.floor(math.log10(rough))
    step = next(factor * magnitude for factor in (1, 2, 2.5, 5, 10)
                if factor * magnitude >= rough)
    lower = math.floor(low / step) * step
    upper = math.ceil(high / step) * step
    if lower == upper:
        upper += step
    return lower, upper, step


def plot(rows, metric, ylabel):
    width, height = 1800, 1000
    left, right, top, bottom = 275, 1720, 155, 780
    values = [(float(rows[(method, traffic)][f"{metric}_paired_delta_mean"]),
               float(rows[(method, traffic)][f"{metric}_paired_delta_sd"]))
              for method, _, _ in METHODS for traffic in TRAFFIC]
    ymin, ymax, step = nice_axis(values)

    def ymap(value):
        return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

    text_style = 'font-family="Times New Roman" font-size="29pt" fill="#222"'
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(ylabel)} by method and traffic">',
        '<desc>Cross-region minus within-region, five paired seeds; '
        'error bars show plus or minus one sample standard deviation.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    nticks = round((ymax - ymin) / step)
    for index in range(nticks + 1):
        value = ymin + index * step
        y = ymap(value)
        zero = abs(value) < step * 1e-7
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            f'stroke="{"#858585" if zero else "#e0e0e0"}" '
            f'stroke-width="{"2.5" if zero else "1.5"}"/>'
        )
        tick = f"{value:.0f}" if step >= 1 else f"{value:.1f}"
        parts.append(
            f'<text x="{left-20}" y="{y+13:.1f}" text-anchor="end" '
            f'{text_style}>{escape(tick)}</text>'
        )

    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" '
                 'stroke="#222" stroke-width="2.5"/>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" '
                 'stroke="#222" stroke-width="2.5"/>')

    for method_index, (method, label, color) in enumerate(METHODS):
        center = left + (method_index + 0.5) * (right - left) / len(METHODS)
        parts.append(f'<text x="{center:.1f}" y="{bottom+66}" text-anchor="middle" '
                     f'{text_style}>{escape(label)}</text>')
        for traffic, offset in (("normal", -30), ("ftp", 30)):
            row = rows[(method, traffic)]
            mean = float(row[f"{metric}_paired_delta_mean"])
            sd = float(row[f"{metric}_paired_delta_sd"])
            x, y = center + offset, ymap(mean)
            high, low = ymap(mean + sd), ymap(mean - sd)
            zero = ymap(0)
            bar_top, bar_height = min(y, zero), abs(y - zero)
            opacity = "1" if traffic == "normal" else "0.45"
            parts.extend((
                f'<rect x="{x-23:.1f}" y="{bar_top:.1f}" width="46" '
                f'height="{bar_height:.1f}" fill="{color}" fill-opacity="{opacity}" '
                f'stroke="{color}" stroke-width="2"/>',
                f'<line x1="{x:.1f}" y1="{high:.1f}" x2="{x:.1f}" '
                f'y2="{low:.1f}" stroke="#222" stroke-width="3.5"/>',
                f'<line x1="{x-13:.1f}" y1="{high:.1f}" x2="{x+13:.1f}" '
                f'y2="{high:.1f}" stroke="#222" stroke-width="3.5"/>',
                f'<line x1="{x-13:.1f}" y1="{low:.1f}" x2="{x+13:.1f}" '
                f'y2="{low:.1f}" stroke="#222" stroke-width="3.5"/>',
            ))

    parts.extend((
        '<rect x="522" y="56" width="36" height="30" fill="#555"/>',
        f'<text x="568" y="89" {text_style}>Normal traffic</text>',
        '<rect x="1032" y="56" width="36" height="30" fill="#555" '
        'fill-opacity="0.45" stroke="#555" stroke-width="2"/>',
        f'<text x="1080" y="89" {text_style}>FTP burst</text>',
        f'<text x="85" y="{(top+bottom)/2:.1f}" '
        f'transform="rotate(-90 85 {(top+bottom)/2:.1f})" '
        f'text-anchor="middle" {text_style}>{escape(ylabel)}</text>',
        '</svg>',
    ))
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, stem in METRICS:
        (args.output_dir / f"{stem}.svg").write_text(
            plot(rows, metric, ylabel), encoding="utf-8")


if __name__ == "__main__":
    main()
