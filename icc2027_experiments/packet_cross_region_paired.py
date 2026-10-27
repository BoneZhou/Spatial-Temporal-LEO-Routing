"""Paired same-/cross-region audit for the completed packet-policy holdout."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from .packet_validation import METHODS


PAIRS = ((11, 101), (22, 202), (33, 303), (44, 404), (55, 505))
COMPARISONS = (
    ("normal", "S1_non_cross_normal", "S2_cross_normal"),
    ("ftp", "S3_non_cross_ftp", "S4_cross_ftp"),
)
LABELS = {
    "proposed_gat_lstm_packet": "Proposed",
    "graphpr_adapted": "GraphPR-adapted",
    "fdr_marl_adapted": "FDR-MARL-adapted",
    "dqn_ir_fixed_link": "DQN-IR-adapted (fixed-link)",
}
METRICS = (
    ("throughput_mbps", 1.0, "Mbps"),
    ("packet_loss_rate", 100.0, "percentage points"),
    ("end_to_end_delay_ms", 1.0, "ms"),
    ("average_queue_packets", 1.0, "packets/satellite"),
    ("mean_hops", 1.0, "hops"),
    ("mean_shortest_hops_at_birth", 1.0, "hops"),
    ("mean_inter_plane_crossings", 1.0, "crossings"),
    ("queue_wait_ms", 1.0, "ms"),
    ("propagation_delay_ms", 1.0, "ms"),
    ("delay_per_mean_hop_ms", 1.0, "ms/hop"),
    ("queue_wait_per_mean_hop_ms", 1.0, "ms/hop"),
)


def load_record(root: Path, checkpoint_seed: int, traffic_seed: int,
                method: str, scenario: str, trace_hash: str):
    path = (root / f"checkpoint{checkpoint_seed}_traffic{traffic_seed}" /
            method / scenario / "summary_FORMAL.json")
    row = json.loads(path.read_text(encoding="utf-8"))
    if (row["status"] != "FORMAL_COMPLETE" or row["checkpoint_seed"] != checkpoint_seed or
            row["traffic_seed"] != traffic_seed or row["variant"] != method or
            row["scenario"] != scenario or row["trace_sha256"] != trace_hash or
            row["background_load_mbps"] != 240.0 or
            row["generated_packets"] != row["delivered_packets"] +
            row["dropped_packets"] + row["in_flight_at_horizon"]):
        raise RuntimeError(f"Invalid formal record: {path}")
    if row["mean_hops"] is None or row["mean_hops"] <= 0:
        raise RuntimeError(f"Undefined delivered-packet mean hops: {path}")
    row["delay_per_mean_hop_ms"] = row["end_to_end_delay_ms"] / row["mean_hops"]
    row["queue_wait_per_mean_hop_ms"] = row["queue_wait_ms"] / row["mean_hops"]
    return row


def mean_sd(values):
    if len(values) != 5 or any(not math.isfinite(value) for value in values):
        raise RuntimeError("Expected five finite paired values")
    return statistics.mean(values), statistics.stdev(values)


def render(value):
    return f"{value:+.2f}"


def paired_analysis(root: Path):
    if not (root / "FORMAL_COMPLETE.ok").is_file():
        raise RuntimeError("Formal evaluation is incomplete")
    audit = json.loads((root / "packet_audit_FORMAL.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PACKET_AUDIT_PASSED" or audit.get("scenarios") != 80:
        raise RuntimeError("All 80 packet histories must pass audit")
    seed_rows = []
    recovery = {}
    for checkpoint_seed, traffic_seed in PAIRS:
        pair_root = root / f"checkpoint{checkpoint_seed}_traffic{traffic_seed}"
        manifest = json.loads((pair_root / "trace_manifest.json").read_text(encoding="utf-8"))
        if (manifest["checkpoint_seed"] != checkpoint_seed or
                manifest["traffic_seed"] != traffic_seed or
                manifest["settings"]["background_load_mbps"] != 240.0):
            raise RuntimeError(f"Invalid trace manifest: {pair_root}")
        trace_hash = manifest["trace_sha256"]
        for method in METHODS:
            for label, same_name, cross_name in COMPARISONS:
                same = load_record(root, checkpoint_seed, traffic_seed, method,
                                   same_name, trace_hash)
                cross = load_record(root, checkpoint_seed, traffic_seed, method,
                                    cross_name, trace_hash)
                if (same["generated_packets"] != cross["generated_packets"] or
                        same["nhpp_injected_packets"] != cross["nhpp_injected_packets"] or
                        same["ftp_injected_packets"] != cross["ftp_injected_packets"] or
                        same["measurement_slots"] != cross["measurement_slots"]):
                    raise RuntimeError(f"Region pair received different traffic: {method}/{label}/{traffic_seed}")
                if label == "ftp":
                    recovery[(method, traffic_seed)] = (same["recovery_status"],
                                                        cross["recovery_status"])
                row = {
                    "checkpoint_seed": checkpoint_seed, "traffic_seed": traffic_seed,
                    "method": method, "traffic": label,
                    "generated_packets": same["generated_packets"],
                    "trace_sha256": trace_hash,
                }
                for metric, scale, _ in METRICS:
                    same_value = float(same[metric]) * scale
                    cross_value = float(cross[metric]) * scale
                    row[f"{metric}_same"] = same_value
                    row[f"{metric}_cross"] = cross_value
                    row[f"{metric}_delta"] = cross_value - same_value
                row["throughput_relative_change_percent"] = (
                    100.0 * (cross["throughput_mbps"] / same["throughput_mbps"] - 1.0)
                )
                seed_rows.append(row)
    per_seed_path = root / "paired_cross_region_per_seed_240.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    aggregate_rows = []
    for method in METHODS:
        for label, _, _ in COMPARISONS:
            cohort = [row for row in seed_rows if row["method"] == method and
                      row["traffic"] == label]
            if len(cohort) != 5:
                raise RuntimeError(f"Incomplete cohort: {method}/{label}")
            summary = {"method": method, "traffic": label, "n_pairs": 5}
            for metric, _, _ in METRICS:
                mean, sd = mean_sd([row[f"{metric}_delta"] for row in cohort])
                summary[f"{metric}_paired_delta_mean"] = mean
                summary[f"{metric}_paired_delta_sd"] = sd
            mean, sd = mean_sd([row["throughput_relative_change_percent"]
                                for row in cohort])
            summary["throughput_relative_change_percent_mean"] = mean
            summary["throughput_relative_change_percent_sd"] = sd
            aggregate_rows.append(summary)
    aggregate_path = root / "paired_cross_region_mean_sd_240.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    lines = [
        "# 240 Mbps 逐封包四方法：跨區／不跨區配對分析", "",
        "五組 checkpoint seeds 11／22／33／44／55 分別配對預留流量 seeds 101／202／303／404／505。"
        "同 seed、同方法下以 S2−S1 比較正常流量，以 S4−S3 比較 FTP 流量；外生流量、觀察長度與 trace hash 已逐對核對。"
        "每欄先計算五個配對差值，再報平均值（mean）± 樣本標準差（sample standard deviation），不是用兩組總平均相減來估標準差。",
        "跨區是來源與目的地位於不同 orbital-plane groups；不跨區是位於同一組。兩者都可使用跨平面 ISL 作為可行鏈路。", "",
        "## 四項主指標的配對差值（跨區 − 不跨區）", "",
        "| 方法 | 流量 | 吞吐量（Mbps） | 相對吞吐變化（%） | 損失率（百分點） | 端到端延遲（ms） | 平均佇列（packets/衛星） |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        def pair(metric):
            return (f"{render(row[f'{metric}_paired_delta_mean'])} ± "
                    f"{row[f'{metric}_paired_delta_sd']:.2f}")
        lines.append(
            f"| {LABELS[row['method']]} | {row['traffic']} | "
            f"{pair('throughput_mbps')} | "
            f"{render(row['throughput_relative_change_percent_mean'])} ± "
            f"{row['throughput_relative_change_percent_sd']:.2f} | "
            f"{pair('packet_loss_rate')} | {pair('end_to_end_delay_ms')} | "
            f"{pair('average_queue_packets')} |"
        )
    lines += [
        "", "## 路徑與延遲解釋（僅已交付封包）", "",
        "| 方法 | 流量 | 實際 hops 差 | 最短 hops 差 | 跨平面次數差 | 佇列等待差（ms） | 每平均 hop 延遲差（ms/hop） |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        def pair(metric):
            return (f"{render(row[f'{metric}_paired_delta_mean'])} ± "
                    f"{row[f'{metric}_paired_delta_sd']:.2f}")
        lines.append(
            f"| {LABELS[row['method']]} | {row['traffic']} | "
            f"{pair('mean_hops')} | {pair('mean_shortest_hops_at_birth')} | "
            f"{pair('mean_inter_plane_crossings')} | {pair('queue_wait_ms')} | "
            f"{pair('delay_per_mean_hop_ms')} |"
        )
    lines += [
        "", "每平均 hop 延遲＝該 seed 已交付封包的平均端到端延遲／平均實際 hops，**不是**逐封包延遲/hop 的平均；兩組交付封包集合也可能不同。"
        "跨區的 raw delay 增加同時包含路徑變長與等待變化，不能只歸因於壅塞。高損失方法的延遲只描述成功交付的子集合，須連同損失率閱讀。",
        "五個 seeds 的誤差棒不等於顯著性檢定（significance test）；不據此宣稱方法普遍優於另一方法。", "",
        "## 120／240 Mbps 在論文中的分工", "",
        "- **240 Mbps，新逐封包公平主比較**：Proposed、GraphPR-adapted、FDR-MARL-adapted、DQN-IR-adapted 共用訓練／評估介面與五組配對 seeds。四項主指標及本報告的跨區效應可放主結果。S3/S4 是在 **240 Mbps NHPP 背景上額外加入 FTP**，絕不能標成 120 Mbps 或『總負載 240 Mbps』。",
        "- **120 Mbps，舊四消融的 FTP robustness**：DQN、LSTM-DQN、GAT-DQN、GAT-LSTM-DQN 使用 240 Mbps 訓練的舊單焦點介面 checkpoint，在 **120 Mbps NHPP 背景上額外加入 FTP**；已有時間曲線與恢復分類。最多作獨立探索性／補充結果；其 Proposed 不是上述逐封包主比較的同一個訓練介面、checkpoint 或評估口徑。兩套 Proposed 數字不能拼成同一條 load curve、單一模型排名，也不能用舊消融直接證明新版 Proposed 的 GAT／LSTM 個別貢獻。若稿件要嚴格滿足最終方法的消融要求，須另在同一逐封包協定下完成四變體對照；本分析不啟動重訓。",
        "- 若論文主張『同一組四個逐封包方法在 120 Mbps FTP 下的恢復能力』，現有資料**不支持**；需另以已凍結 checkpoint 做 120 Mbps 的獨立配對評估（不用重訓）。在決定要放這項主張前，不自行啟動新測試。",
        "", "本分析只讀取已完成的正式輸出；80／80 場、35,045,296 筆封包通過既有稽核。每 seed 原始配對值與完整精度的平均／標準差分別見 `paired_cross_region_per_seed_240.csv`、`paired_cross_region_mean_sd_240.csv`。", "",
    ]
    report_path = root / "PAIRED_CROSS_REGION_zh.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return per_seed_path, aggregate_path, report_path, recovery


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = paired_analysis(args.output_dir.resolve())
    print(f"Wrote {result[0].name}, {result[1].name}, {result[2].name}")
