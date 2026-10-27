"""Combine frozen full-Proposed and packet-level ablation formal evaluations."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from .robustness_pilot import SCENARIOS


PAIRS = ((11, 101), (22, 202), (33, 303), (44, 404), (55, 505))
METHODS = (
    "proposed_gat_lstm_packet", "proposed_gat_dqn_packet",
    "proposed_lstm_dqn_packet", "proposed_dqn_packet",
)
LABELS = {
    "proposed_gat_lstm_packet": "GAT + LSTM (full)",
    "proposed_gat_dqn_packet": "GAT only",
    "proposed_lstm_dqn_packet": "LSTM only",
    "proposed_dqn_packet": "Neither (DQN)",
}
METRICS = (
    ("throughput_mbps", "吞吐量（Mbps）", 1.0),
    ("packet_loss_rate", "封包損失率（%）", 100.0),
    ("end_to_end_delay_ms", "端到端延遲（ms）", 1.0),
    ("average_queue_packets", "平均佇列長度（packets/衛星）", 1.0),
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(project: Path, output: Path, *, full_root=None, new_root=None) -> None:
    full_root = full_root or project / "outputs" / "packet_fair_formal_load240_fiveseed"
    new_root = new_root or project / "outputs" / "packet_ablation_formal_load240_fiveseed"
    if not (full_root / "FORMAL_COMPLETE.ok").is_file() or \
            not (new_root / "FORMAL_COMPLETE.ok").is_file():
        raise RuntimeError("Full and new ablation formal evaluations must both complete")
    full_audit = _read_json(full_root / "packet_audit_FORMAL.json")
    new_audit = _read_json(new_root / "packet_audit_FORMAL.json")
    if (full_audit.get("status") != "PACKET_AUDIT_PASSED" or
            full_audit.get("scenarios") != 80 or
            new_audit.get("status") != "PACKET_AUDIT_PASSED" or
            new_audit.get("scenarios") != 60):
        raise RuntimeError("Both per-packet formal audits must pass before aggregation")
    records = {}
    for seed, traffic_seed in PAIRS:
        pair = f"checkpoint{seed}_traffic{traffic_seed}"
        old_trace = _read_json(full_root / pair / "trace_manifest.json")
        new_trace = _read_json(new_root / pair / "trace_manifest.json")
        if old_trace != new_trace:
            raise RuntimeError(f"Formal exogenous trace differs: {pair}")
        for scenario, _, _ in SCENARIOS:
            for method in METHODS:
                root = full_root if method == METHODS[0] else new_root
                path = root / pair / method / scenario / "summary_FORMAL.json"
                row = _read_json(path)
                if (row["status"] != "FORMAL_COMPLETE" or
                        row["checkpoint_seed"] != seed or
                        row["traffic_seed"] != traffic_seed or
                        row["trace_sha256"] != old_trace["trace_sha256"] or
                        row["generated_packets"] != row["delivered_packets"] +
                        row["dropped_packets"] + row["in_flight_at_horizon"]):
                    raise RuntimeError(f"Invalid formal row: {path}")
                for metric, _, _ in METRICS:
                    if row[metric] is None or not math.isfinite(float(row[metric])):
                        raise RuntimeError(f"Undefined metric: {path}, {metric}")
                records[(seed, method, scenario)] = row
            generated = {records[(seed, method, scenario)]["generated_packets"]
                         for method in METHODS}
            if len(generated) != 1:
                raise RuntimeError(f"Unpaired packet injection: {pair}/{scenario}")
    summaries = []
    differences = []
    for scenario, _, _ in SCENARIOS:
        for method in METHODS:
            cohort = [records[(seed, method, scenario)] for seed, _ in PAIRS]
            summary = {"scenario": scenario, "method": method, "n_seeds": len(PAIRS)}
            for metric, _, scale in METRICS:
                values = [float(row[metric]) * scale for row in cohort]
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_sd"] = statistics.stdev(values)
            summaries.append(summary)
            if method == METHODS[0]:
                continue
            for seed, _ in PAIRS:
                full = records[(seed, METHODS[0], scenario)]
                reduced = records[(seed, method, scenario)]
                pair_row = {"scenario": scenario, "seed": seed, "ablated_method": method}
                for metric, _, scale in METRICS:
                    pair_row[f"full_minus_ablated_{metric}"] = (
                        float(full[metric]) - float(reduced[metric])) * scale
                differences.append(pair_row)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "ablation_four_metrics_mean_sd.csv").open(
            "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (output / "ablation_paired_differences.csv").open(
            "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(differences[0]))
        writer.writeheader()
        writer.writerows(differences)
    lines = [
        "# 新版逐封包 Proposed 嚴格消融：240 Mbps 五種子正式結果", "",
        "完整版與三個消融版均以相同逐封包觀測／獎勵／互動預算訓練。",
        "凍結 checkpoint seeds 11／22／33／44／55，配對獨立測試流量 seeds 101／202／303／404／505；epsilon＝0。",
        "完整版沿用原 20 場正式結果；新三變體完成 60 場，兩批流量 manifest 完全一致。",
        "每方法／場景 n＝5；表內為平均值（mean）± 樣本標準差（sample standard deviation）。",
        "末段學習曲線不能證明所有變體均收斂，故本表只稱等互動預算消融（fixed-interaction-budget ablation）。", "",
        "| 架構 | 場景 | 吞吐量（Mbps） | 封包損失率（%） | 端到端延遲（ms） | 平均佇列長度（packets/衛星） |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        values = [f"{row[f'{metric}_mean']:.2f} ± {row[f'{metric}_sd']:.2f}"
                  for metric, _, _ in METRICS]
        lines.append(f"| {LABELS[row['method']]} | {row['scenario']} | " +
                     " | ".join(values) + " |")
    lines += [
        "", "## 配對差值", "",
        "以下為**完整版 − 消融版**的同 seed 差值，計算前先以相同外生流量配對；正值不一定代表改善，須依指標方向解讀。完整每 seed 差值見 `ablation_paired_differences.csv`。", "",
        "| 消融版 | 場景 | 吞吐量差（Mbps） | 損失率差（百分點） | 延遲差（ms） | 佇列差（packets/衛星） |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scenario, _, _ in SCENARIOS:
        for method in METHODS[1:]:
            cohort = [row for row in differences if row["scenario"] == scenario and
                      row["ablated_method"] == method]
            values = []
            for metric, _, _ in METRICS:
                field = f"full_minus_ablated_{metric}"
                sample = [row[field] for row in cohort]
                values.append(f"{statistics.mean(sample):+.2f} ± {statistics.stdev(sample):.2f}")
            lines.append(f"| {LABELS[method]} | {scenario} | " +
                         " | ".join(values) + " |")
    lines += [
        "", "兩批逐封包稽核均通過：既有四方法正式評估 80 場，新消融 60 場。",
        "延遲只對已交付封包計算，應連同損失率解讀；尚未到達量測終點的封包不算丟棄。",
        "這些差值是相同負載／預算下的模組消融描述，非跨負載因果推論；不把舊單焦點消融結果併入。", "",
    ]
    (output / "ABLATION_FOUR_METRICS_zh.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.project.resolve(), args.output.resolve())
