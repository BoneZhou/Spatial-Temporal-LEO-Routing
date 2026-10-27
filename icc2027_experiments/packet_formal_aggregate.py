"""Summarize only the four prespecified packet-routing performance metrics."""

import argparse
import csv
import json
import math
import statistics
from html import escape
from pathlib import Path

from .packet_validation import METHODS
from .robustness_pilot import SCENARIOS


PAIRS = ((11, 101), (22, 202), (33, 303), (44, 404), (55, 505))
METRICS = (
    ("throughput_mbps", "Throughput (Mbps)", "吞吐量（Mbps）", 1.0),
    ("packet_loss_rate", "Packet loss rate (%)", "封包損失率（%）", 100.0),
    ("end_to_end_delay_ms", "End-to-end delay (ms)", "端到端延遲（ms）", 1.0),
    ("average_queue_packets", "Average queue length (packets/satellite)",
     "平均佇列長度（packets/衛星）", 1.0),
)
LABELS = {
    "proposed_gat_lstm_packet": "Proposed",
    "graphpr_adapted": "GraphPR-adapted",
    "fdr_marl_adapted": "FDR-MARL-adapted",
    "dqn_ir_fixed_link": "DQN-IR-adapted (fixed-link)",
}
COLORS = ("#c92f37", "#2374aa", "#42944a", "#8a58a7")


def symbol(x, y, color, index):
    if index == 0:
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>'
    if index == 1:
        return f'<rect x="{x-6:.1f}" y="{y-6:.1f}" width="12" height="12" fill="{color}"/>'
    if index == 2:
        return (f'<polygon points="{x:.1f},{y-7:.1f} {x-7:.1f},{y+6:.1f} '
                f'{x+7:.1f},{y+6:.1f}" fill="{color}"/>')
    return (f'<polygon points="{x:.1f},{y-8:.1f} {x-8:.1f},{y:.1f} '
            f'{x:.1f},{y+8:.1f} {x+8:.1f},{y:.1f}" fill="{color}"/>')


def plot_svg(rows, metric, label):
    width, height = 1200, 720
    left, right, top, bottom = 135, 1150, 95, 610
    ymax = max(float(row[f"{metric}_mean"]) + float(row[f"{metric}_sd"])
               for row in rows) * 1.12
    if ymax <= 0:
        ymax = 1
    def ymap(value):
        return bottom - value / ymax * (bottom - top)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(label)}">',
             '<rect width="100%" height="100%" fill="white"/>']
    for tick in range(6):
        value = ymax * tick / 5
        y = ymap(value)
        parts += [f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
                  'stroke="#dedede" stroke-width="1"/>',
                  f'<text x="{left-12}" y="{y+6:.1f}" text-anchor="end" '
                  f'font-family="Times New Roman" font-size="20">{value:.1f}</text>']
    parts.append(f'<rect x="{left}" y="{top}" width="{right-left}" '
                 f'height="{bottom-top}" fill="none" stroke="#333" stroke-width="1.3"/>')
    scenarios = [row[0] for row in SCENARIOS]
    for scenario_index, scenario in enumerate(scenarios):
        center = left + (scenario_index + 0.5) * (right-left) / 4
        parts.append(f'<text x="{center:.1f}" y="{bottom+34}" text-anchor="middle" '
                     f'font-family="Times New Roman" font-size="21">S{scenario_index+1}</text>')
        for method_index, method in enumerate(METHODS):
            row = next(item for item in rows if item["scenario"] == scenario and
                       item["method"] == method)
            x = center + (method_index - 1.5) * 37
            mean = float(row[f"{metric}_mean"])
            sd = float(row[f"{metric}_sd"])
            high, low = ymap(mean + sd), ymap(max(0.0, mean - sd))
            color = COLORS[method_index]
            parts += [f'<line x1="{x:.1f}" y1="{high:.1f}" x2="{x:.1f}" y2="{low:.1f}" '
                      f'stroke="{color}" stroke-width="2.5"/>']
            for cap in (high, low):
                parts.append(f'<line x1="{x-10:.1f}" y1="{cap:.1f}" '
                             f'x2="{x+10:.1f}" y2="{cap:.1f}" '
                             f'stroke="{color}" stroke-width="2.5"/>')
            parts.append(symbol(x, ymap(mean), color, method_index))
    for index, method in enumerate(METHODS):
        x = 135 + index * 250
        parts.append(symbol(x, 44, COLORS[index], index))
        parts.append(f'<text x="{x+16}" y="51" font-family="Times New Roman" '
                     f'font-size="20">{LABELS[method]}</text>')
    parts += [f'<text x="{(left+right)/2:.1f}" y="{height-32}" text-anchor="middle" '
              'font-family="Times New Roman" font-size="22">Scenario</text>',
              f'<text x="36" y="{(top+bottom)/2:.1f}" '
              f'transform="rotate(-90 36 {(top+bottom)/2:.1f})" text-anchor="middle" '
              f'font-family="Times New Roman" font-size="22">{escape(label)}</text>',
              '</svg>']
    return "\n".join(parts) + "\n"


def aggregate(root: Path):
    if not (root / "FORMAL_COMPLETE.ok").is_file():
        raise RuntimeError("Formal evaluation has not completed")
    audit = json.loads((root / "packet_audit_FORMAL.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PACKET_AUDIT_PASSED" or audit.get("scenarios") != 80:
        raise RuntimeError("All 80 packet-level histories must pass audit")
    records = {}
    for checkpoint_seed, traffic_seed in PAIRS:
        manifest = json.loads((root / f"checkpoint{checkpoint_seed}_traffic{traffic_seed}" /
                               "trace_manifest.json").read_text(encoding="utf-8"))
        for method in METHODS:
            for scenario, _, _ in SCENARIOS:
                path = (root / f"checkpoint{checkpoint_seed}_traffic{traffic_seed}" /
                        method / scenario / "summary_FORMAL.json")
                row = json.loads(path.read_text(encoding="utf-8"))
                if (row["checkpoint_seed"] != checkpoint_seed or
                        row["traffic_seed"] != traffic_seed or
                        row["trace_sha256"] != manifest["trace_sha256"] or
                        row["generated_packets"] != row["delivered_packets"] +
                        row["dropped_packets"] + row["in_flight_at_horizon"]):
                    raise RuntimeError(f"Invalid pair or packet conservation: {path}")
                records[(checkpoint_seed, method, scenario)] = row
    output = []
    for scenario, _, _ in SCENARIOS:
        for method in METHODS:
            cohort = [records[(seed, method, scenario)] for seed, _ in PAIRS]
            row = {"scenario": scenario, "method": method, "n_seeds": 5}
            for key, _, _, scale in METRICS:
                values = [float(item[key]) * scale for item in cohort]
                if any(not math.isfinite(value) for value in values):
                    raise RuntimeError(f"Nonfinite metric: {method}/{scenario}/{key}")
                row[f"{key}_mean"] = statistics.mean(values)
                row[f"{key}_sd"] = statistics.stdev(values)
            output.append(row)
    with (root / "four_metrics_mean_sd.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    lines = ["# 240 Mbps 五個 seeds 正式四項指標評估", "",
             "四方法均使用凍結的逐封包 checkpoint（11、22、33、44、55），與未參與訓練的配對流量 seeds（101、202、303、404、505）。探索率（epsilon）＝0。",
             "每個場景先計算各 seed 的指標，再取五組的平均值（mean）± 樣本標準差（sample standard deviation）；誤差棒（error bars）為 ±1 SD。",
             "結果屬於**等互動預算比較**，不宣稱四方法都已充分收斂。", "",
             "| 方法 | 場景 | 吞吐量（Mbps） | 封包損失率（%） | 端到端延遲（ms） | 平均佇列長度（packets/衛星） |",
             "|---|---|---:|---:|---:|---:|"]
    for row in output:
        formatted = [f"{row[f'{key}_mean']:.2f} ± {row[f'{key}_sd']:.2f}"
                     for key, _, _, _ in METRICS]
        lines.append(f"| {LABELS[row['method']]} | {row['scenario']} | " +
                     " | ".join(formatted) + " |")
    lines += ["", "封包損失率（packet loss rate）＝已丟棄／已產生；觀察終點仍在途中的封包不算損失。端到端延遲（end-to-end delay）只對已交付封包計算，須連同損失率閱讀。",
              "平均佇列長度（average queue length）是觀察時段內逐時槽、逐衛星共用佇列長度的平均。",
              f"所有 80 場封包軌跡已通過逐封包稽核，共 {audit['packet_rows']:,} 筆。", ""]
    (root / "FOUR_METRICS_REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")
    figures = root / "four_metric_figures"
    figures.mkdir(exist_ok=True)
    for key, label, _, _ in METRICS:
        (figures / f"{key}_errorbars.svg").write_text(
            plot_svg(output, key, label), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    aggregate(args.output_dir.resolve())
