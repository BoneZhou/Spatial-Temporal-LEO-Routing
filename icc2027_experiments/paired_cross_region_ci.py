"""Compute paired, unadjusted 95% t intervals from five held-out seed pairs."""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


SEED_PAIRS = {11: 101, 22: 202, 33: 303, 44: 404, 55: 505}
METHODS = (
    ("proposed_gat_lstm_packet", "Proposed"),
    ("graphpr_adapted", "GraphPR-adapted"),
    ("fdr_marl_adapted", "FDR-MARL-adapted"),
    ("dqn_ir_fixed_link", "DQN-IR-adapted (fixed-link)"),
)
TRAFFIC = ("normal", "ftp")
METRICS = (
    ("throughput_mbps", "吞吐量（Mbps）"),
    ("packet_loss_rate", "封包損失率（百分點）"),
    ("end_to_end_delay_ms", "端到端延遲（ms）"),
    ("average_queue_packets", "平均佇列長度（packets/衛星）"),
)
T_CRITICAL_95_DF4 = 2.7764451051977987


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def analyze(per_seed_path, summary_path):
    per_seed = read_csv(per_seed_path)
    summary = read_csv(summary_path)
    if len(per_seed) != 40 or len(summary) != 8:
        raise ValueError("Expected 40 paired records and 8 summary records")
    summaries = {(row["method"], row["traffic"]): row for row in summary}
    if len(summaries) != 8:
        raise ValueError("Duplicate method/traffic summary")

    groups = defaultdict(list)
    seen = set()
    for row in per_seed:
        seed = int(row["checkpoint_seed"])
        traffic_seed = int(row["traffic_seed"])
        key = (row["method"], row["traffic"])
        record_key = (seed, *key)
        if seed not in SEED_PAIRS or traffic_seed != SEED_PAIRS[seed]:
            raise ValueError(f"Unexpected checkpoint/traffic seed pair: {record_key}")
        if record_key in seen:
            raise ValueError(f"Duplicate paired record: {record_key}")
        seen.add(record_key)
        for metric, _ in METRICS:
            same = float(row[f"{metric}_same"])
            cross = float(row[f"{metric}_cross"])
            delta = float(row[f"{metric}_delta"])
            if not all(math.isfinite(value) for value in (same, cross, delta)):
                raise ValueError(f"Nonfinite {metric}: {record_key}")
            if not math.isclose(cross - same, delta, rel_tol=1e-10, abs_tol=1e-9):
                raise ValueError(f"Paired difference mismatch: {record_key}/{metric}")
        groups[key].append(row)

    expected = {(method, traffic) for method, _ in METHODS for traffic in TRAFFIC}
    if set(groups) != expected or set(summaries) != expected:
        raise ValueError("Missing or unexpected method/traffic combination")

    results = []
    for metric, label in METRICS:
        for method, display in METHODS:
            for traffic in TRAFFIC:
                key = (method, traffic)
                records = groups[key]
                if {int(row["checkpoint_seed"]) for row in records} != set(SEED_PAIRS):
                    raise ValueError(f"Missing seed for {key}")
                if int(summaries[key]["n_pairs"]) != 5:
                    raise ValueError(f"Summary pair count mismatch: {key}")
                values = [float(row[f"{metric}_delta"]) for row in records]
                mean = statistics.mean(values)
                sd = statistics.stdev(values)
                previous_mean = float(summaries[key][f"{metric}_paired_delta_mean"])
                previous_sd = float(summaries[key][f"{metric}_paired_delta_sd"])
                if not (math.isclose(mean, previous_mean, rel_tol=1e-10, abs_tol=1e-9)
                        and math.isclose(sd, previous_sd, rel_tol=1e-10, abs_tol=1e-9)):
                    raise ValueError(f"Summary mean/SD mismatch: {key}/{metric}")
                half_width = T_CRITICAL_95_DF4 * sd / math.sqrt(len(values))
                results.append({
                    "metric": metric,
                    "metric_label_zh": label,
                    "method": method,
                    "method_label": display,
                    "traffic": traffic,
                    "n_pairs": len(values),
                    "mean": mean,
                    "sample_sd": sd,
                    "ci95_lower": mean - half_width,
                    "ci95_upper": mean + half_width,
                    "ci95_half_width": half_width,
                })
    return results


def write_outputs(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "paired_cross_region_95ci.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    lines = [
        "# 240 Mbps 跨區／不跨區配對差值的 95% 信賴區間",
        "",
        "差值定義為「跨區 − 不跨區」。正常流量（normal）是 240 Mbps NHPP 背景、未加入 FTP burst；FTP 場景是在同一背景上額外加入 burst。",
        "使用五組 checkpoint／預留流量 seeds：11／101、22／202、33／303、44／404、55／505。每組先計算同方法、同 seed 的差值，再對五個差值估計平均值（mean）、樣本標準差（sample standard deviation）與 95% 信賴區間（95% confidence interval, CI）。",
        "",
        "公式：95% CI = mean ± t(0.975, 4) × SD / √5；t(0.975, 4) = 2.776445。這是**未作多重比較校正的雙側 Student t 區間**，不是先前圖上的 ±1 SD 誤差棒（error bars）。",
        "",
    ]
    for metric, label in METRICS:
        lines.extend((
            f"## {label}",
            "",
            "| 方法 | 流量 | 配對平均值 ± SD | 95% CI |",
            "|---|---|---:|---:|",
        ))
        for row in results:
            if row["metric"] != metric:
                continue
            lines.append(
                f'| {row["method_label"]} | {row["traffic"]} | '
                f'{row["mean"]:+.2f} ± {row["sample_sd"]:.2f} | '
                f'[{row["ci95_lower"]:+.2f}, {row["ci95_upper"]:+.2f}] |'
            )
        lines.append("")
    lines.extend((
        "## 解讀限制",
        "",
        "- 每組只有 5 個配對 seeds。Student t 區間假設配對差值彼此近似獨立，且母體差值分布可用常態模型近似；樣本太少，無法可靠驗證常態性。",
        "- 各方法、流量與指標的區間是逐項 95% CI，**沒有**對多重比較（multiple comparisons）調整。區間是否涵蓋 0 不等於已完成跨方法優劣或一般化顯著性檢定（significance test）。",
        "- 這些區間只描述目前 240 Mbps、既定星座／佇列模型／來源目的地抽樣規則下的平均配對差值；不能外推至其他負載或設定。",
        "- 端到端延遲只對成功交付封包計算；跨區與不跨區的已交付封包集合可能不同，須與損失率一起閱讀。",
        "",
        "資料來源：上層資料夾的 `paired_cross_region_per_seed_240.csv`。與 `paired_cross_region_mean_sd_240.csv` 的 32 組平均值與 SD 逐項核對一致。",
        "",
    ))
    (output_dir / "PAIRED_CROSS_REGION_95CI_zh.md").write_text(
        "\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-seed", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_outputs(analyze(args.per_seed, args.summary), args.output_dir)


if __name__ == "__main__":
    main()
