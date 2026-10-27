"""Aggregate completed four-scenario holdout evaluation by independent seed."""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .robustness_pilot import SCENARIOS


VARIANTS = ("dqn", "lstm_dqn", "gat_dqn", "gat_lstm_dqn")
DISPLAY = {
    "dqn": "DQN",
    "lstm_dqn": "LSTM-DQN",
    "gat_dqn": "GAT-DQN",
    "gat_lstm_dqn": "GAT-LSTM-DQN（Proposed）",
}
SEED_PAIRS = ((11, 101), (22, 202), (33, 303), (44, 404), (55, 505))
METRICS = (
    ("throughput_mbps", "Mbps"),
    ("packet_loss_rate", "fraction"),
    ("end_to_end_delay_ms", "ms"),
    ("average_queue_packets", "packets/satellite"),
    ("mean_hops", "hops"),
    ("mean_shortest_hops_at_birth", "hops"),
    ("mean_inter_plane_crossings", "crossings"),
    ("queue_wait_ms", "ms"),
    ("propagation_delay_ms", "ms"),
    ("in_flight_fraction", "fraction"),
)


def mean_sd(values):
    if len(values) != len(SEED_PAIRS):
        raise RuntimeError(f"Expected five independent seeds, got {len(values)}")
    if any(value is None for value in values):
        return None, None
    return statistics.mean(values), statistics.stdev(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if not (root / "FORMAL_COMPLETE.ok").exists():
        raise RuntimeError("Formal evaluation has not finished; refusing partial aggregation")
    audit_path = root / "packet_audit_FORMAL.json"
    if not audit_path.exists():
        raise RuntimeError("Packet-level audit must pass before aggregation")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PACKET_AUDIT_PASSED" or audit.get("scenarios") != 80:
        raise RuntimeError("Packet-level audit did not cover all 80 scenarios")
    expected = len(SEED_PAIRS) * len(VARIANTS) * len(SCENARIOS)
    records = {}
    for checkpoint_seed, traffic_seed in SEED_PAIRS:
        manifest_path = (root / f"checkpoint{checkpoint_seed}_traffic{traffic_seed}"
                         / "trace_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for variant in VARIANTS:
            for scenario, _, _ in SCENARIOS:
                path = (root / f"checkpoint{checkpoint_seed}_traffic{traffic_seed}"
                        / variant / scenario / "summary_FORMAL.json")
                record = json.loads(path.read_text(encoding="utf-8"))
                if record["status"] != "FORMAL_COMPLETE":
                    raise RuntimeError(f"Incomplete result: {path}")
                if record["checkpoint_seed"] != checkpoint_seed or record["traffic_seed"] != traffic_seed:
                    raise RuntimeError(f"Seed mismatch: {path}")
                if record["trace_sha256"] != manifest["trace_sha256"]:
                    raise RuntimeError(f"Trace mismatch: {path}")
                if record["generated_packets"] != (
                    record["delivered_packets"] + record["dropped_packets"]
                    + record["in_flight_at_horizon"]
                ):
                    raise RuntimeError(f"Packet conservation mismatch: {path}")
                records[(checkpoint_seed, traffic_seed, variant, scenario)] = record
    if len(records) != expected:
        raise RuntimeError("Unexpected number of scenario summaries")
    for checkpoint_seed, traffic_seed in SEED_PAIRS:
        for scenario, _, _ in SCENARIOS:
            if len({records[(checkpoint_seed, traffic_seed, variant, scenario)]["generated_packets"]
                    for variant in VARIANTS}) != 1:
                raise RuntimeError("Generated packet counts differ across variants")
        for variant in VARIANTS:
            own = {scenario: records[(checkpoint_seed, traffic_seed, variant, scenario)]
                   for scenario, _, _ in SCENARIOS}
            for left, right in (("S1_non_cross_normal", "S2_cross_normal"),
                                ("S3_non_cross_ftp", "S4_cross_ftp")):
                if own[left]["generated_packets"] != own[right]["generated_packets"]:
                    raise RuntimeError("Paired region scenarios have unequal injection")
    rows = []
    for variant in VARIANTS:
        for scenario, _, _ in SCENARIOS:
            cohort = [records[(checkpoint_seed, traffic_seed, variant, scenario)]
                      for checkpoint_seed, traffic_seed in SEED_PAIRS]
            row = {"variant": variant, "scenario": scenario, "n_seeds": len(cohort)}
            for metric, _unit in METRICS:
                row[f"{metric}_mean"], row[f"{metric}_sd"] = mean_sd(
                    [record[metric] for record in cohort]
                )
            statuses = Counter(record["recovery_status"] for record in cohort)
            row["recovery_status_counts"] = json.dumps(statuses, ensure_ascii=False)
            recovered = [record["recovery_time_s"] for record in cohort
                         if record["recovery_status"] == "recovered"]
            row["recovered_n"] = len(recovered)
            row["recovery_time_s_recovered_only_mean"] = statistics.mean(recovered) if recovered else None
            row["recovery_time_s_recovered_only_sd"] = statistics.stdev(recovered) if len(recovered) > 1 else None
            rows.append(row)
    with (root / "five_seed_four_scenario_mean_sd.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paired_effects = []
    for variant in VARIANTS:
        for label, same, cross in (
            ("cross_minus_same_normal", "S1_non_cross_normal", "S2_cross_normal"),
            ("cross_minus_same_ftp", "S3_non_cross_ftp", "S4_cross_ftp"),
        ):
            effect = {"variant": variant, "comparison": label}
            for metric in ("throughput_mbps", "packet_loss_rate", "end_to_end_delay_ms",
                           "average_queue_packets", "mean_hops", "queue_wait_ms"):
                deltas = []
                for checkpoint_seed, traffic_seed in SEED_PAIRS:
                    left = records[(checkpoint_seed, traffic_seed, variant, same)]
                    right = records[(checkpoint_seed, traffic_seed, variant, cross)]
                    deltas.append(right[metric] - left[metric])
                effect[f"{metric}_paired_delta_mean"], effect[f"{metric}_paired_delta_sd"] = mean_sd(deltas)
            per_hop_deltas = []
            for checkpoint_seed, traffic_seed in SEED_PAIRS:
                left = records[(checkpoint_seed, traffic_seed, variant, same)]
                right = records[(checkpoint_seed, traffic_seed, variant, cross)]
                per_hop_deltas.append(
                    right["end_to_end_delay_ms"] / right["mean_hops"]
                    - left["end_to_end_delay_ms"] / left["mean_hops"]
                )
            effect["delay_per_mean_hop_ms_paired_delta_mean"], effect[
                "delay_per_mean_hop_ms_paired_delta_sd"
            ] = mean_sd(per_hop_deltas)
            paired_effects.append(effect)
    with (root / "paired_cross_region_effects_mean_sd.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_effects[0]))
        writer.writeheader()
        writer.writerows(paired_effects)
    lines = [
        "# 跨區與 FTP burst：五組獨立流量 seeds 正式評估",
        "",
        "- 240 Mbps 已凍結 checkpoint seeds：11、22、33、44、55；獨立保留流量 seeds：101、202、303、404、505。每組 checkpoint 對應一組未用於訓練的流量 seed；同組的四模型／四場景共用配對 trace。",
        "- 每個場景 30 slots 預熱、250 slots 正常期、250 slots burst 期、FTP 注入尾段及最多 500 slots 恢復觀察。S1/S2 沒有 FTP，但與 S3/S4 使用相同觀察終點。",
        "- 下表各值為每 seed 先計算一個結果，再取跨 seed 平均值（mean）± 樣本標準差（sample standard deviation）；不是把所有封包直接合併後估計誤差棒。",
        "- Loss 定義為已丟棄／已產生；觀察終點仍在網路中的封包另列右設限（right-censored），不當作丟失。Delay 僅計已交付封包，應連同 loss 閱讀。",
        "- 端到端延遲（end-to-end delay）為 `(delivery_slot − birth_slot + 1) × 20 ms`；傳播、傳輸及處理時間另報診斷值，不重複加到 slot-sojourn delay。此口徑與既有焦點封包延遲不同，不可直接比較兩張表。",
        "", "## 實際注入負載", "",
        "FTP session 數與流量由 Poisson 抽樣決定；下表為 S3/S4 共用的預生成 trace，**不是**把理論上的 180 Mbps 當作每次實測。各階段封包數含 NHPP 與 FTP，正常對照場景則不含 FTP。",
        "",
        "| 保留流量 seed | FTP sessions | 正常期封包 | Burst 期封包 | 檔案尾段封包 | 恢復期封包 | Burst 期實際注入負載（Mbps） |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint_seed, traffic_seed in SEED_PAIRS:
        record = records[(checkpoint_seed, traffic_seed, "dqn", "S3_non_cross_ftp")]
        phases = record["birth_phase_counts"]
        burst_packets = phases["burst"]["generated"]
        lines.append(
            f"| {traffic_seed} | {record['ftp_sessions']} | "
            f"{phases['baseline']['generated']} | {burst_packets} | "
            f"{phases.get('ftp_tail', {}).get('generated', 0)} | "
            f"{phases['recovery']['generated']} | "
            f"{burst_packets * 1500 * 8 / 5 / 1_000_000:.2f} |"
        )
    lines += [
        "", "## 四項主指標", "",
        "| 模型 | 場景 | 吞吐量（Mbps） | 損失率（%） | 端到端延遲（ms） | 平均佇列（packets/衛星） |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        def display(key, scale=1.0):
            mean = row[f"{key}_mean"]
            sd = row[f"{key}_sd"]
            return "NA" if mean is None else f"{mean*scale:.2f} ± {sd*scale:.2f}"
        lines.append(
            f"| {DISPLAY[row['variant']]} | {row['scenario']} | "
            f"{display('throughput_mbps')} | {display('packet_loss_rate', 100)} | "
            f"{display('end_to_end_delay_ms')} | {display('average_queue_packets')} |"
        )
    lines += [
        "", "## 配對跨區效應", "",
        "以下為**同一 seed** 的跨區值減同區值，再對五個配對差值取平均 ± 樣本標準差。每跳延遲欄是 `平均端到端延遲／平均實際 hops` 的配對差，可輔助辨識單純因路徑變長造成的 raw delay 增加；它不是逐封包 per-hop 延遲的平均。",
        "",
        "| 模型 | 流量 | 吞吐量差（Mbps） | 損失率差（百分點） | 原始延遲差（ms） | 每平均 hop 延遲差（ms/hop） |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for effect in paired_effects:
        def delta(key, scale=1.0):
            return (f"{effect[f'{key}_paired_delta_mean']*scale:+.2f} ± "
                    f"{effect[f'{key}_paired_delta_sd']*scale:.2f}")
        lines.append(
            f"| {DISPLAY[effect['variant']]} | "
            f"{'正常' if effect['comparison'].endswith('normal') else 'FTP'} | "
            f"{delta('throughput_mbps')} | {delta('packet_loss_rate', 100)} | "
            f"{delta('end_to_end_delay_ms')} | "
            f"{delta('delay_per_mean_hop_ms')} |"
        )
    lines += ["", "## 恢復狀態", ""]
    for row in rows:
        if row["scenario"].endswith("ftp"):
            lines.append(
                f"- {DISPLAY[row['variant']]}／{row['scenario']}："
                f"{row['recovery_status_counts']}；僅已恢復個案的平均時間="
                f"{row['recovery_time_s_recovered_only_mean']} s（n={row['recovered_n']}）。"
            )
    lines += [
        "", "[恢復時間圖](time_curves/recovery_time.png)中的點代表各個已恢復 seed；橫線及誤差棒為這些個案的平均值（mean）± 樣本標準差（sample standard deviation）。「未出現可辨識 queue 偏離」沒有 recovery time，不能當作 0 秒，也不會出現在圖上。",
    ]
    lines += [
        "", "## 時間曲線", "",
        "曲線以 burst 起點為 0 秒，畫五個 seeds 的逐時點平均；S1／S2 正常場景保留相同的對照時間軸。Queue 為 20-slot 移動平均；loss 為截至該時點的已觀測累計丟棄／累計生成比例（右設限封包尚未算入丟失）；delay 是最近 20 slots 成功送達封包的加權平均。FTP 的檔案注入尾段長度因 seed 而異，圖只畫五 seeds 共同涵蓋的時間範圍。",
        "",
    ]
    for variant in VARIANTS:
        lines.append(f"- {DISPLAY[variant]}："
                     f"[queue](time_curves/{variant}_queue_time.png)、"
                     f"[loss](time_curves/{variant}_loss_time.png)、"
                     f"[delay](time_curves/{variant}_delay_time.png)")
    lines += [
        "", "## 解讀界線", "",
        "- 五個 seeds 的配對結果顯示四模型在跨區場景的吞吐量（throughput）均下降、封包損失率（packet loss rate）均上升；原始端到端延遲（end-to-end delay）也增加，但這包含路徑 hops 增加，不能只歸因於 queue congestion。",
        "- Proposed 在跨區兩場景的平均 throughput 高於 DQN、平均 loss 低於 DQN，但逐 seed 配對差值有正有負，並非每個 seed 都較好。本表尚未做顯著性檢定（significance test），不能宣稱已證明 Proposed 優於 DQN。",
        "- Proposed 的同區 FTP 恢復時間平均 2.72 s，受個別較長恢復個案影響；不可據此宣稱恢復能力優於其他模型。部分場景根本沒有足夠 queue 偏離，這與「快速恢復」是不同分類。",
        f"- 逐封包路徑、出生／交付／丟棄時間、逐 slot 原始數值及各 seed 摘要均保存在同一輸出資料夾；全部 80 組與 {audit['packet_rows']:,} 筆封包已通過逐筆稽核。",
    ]
    (root / "five_seed_four_scenario_report_zh.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Aggregated {expected} scenario summaries in {root}")


if __name__ == "__main__":
    main()
