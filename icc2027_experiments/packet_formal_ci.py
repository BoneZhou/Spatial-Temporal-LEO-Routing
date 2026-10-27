"""Compute unadjusted 95% t intervals for the four formal packet-routing metrics."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from .packet_formal_aggregate import LABELS, METRICS, PAIRS
from .packet_validation import METHODS
from .robustness_pilot import SCENARIOS


T_CRITICAL_95_DF4 = 2.7764451051977987
EXPECTED_PAIRS = dict(PAIRS)
SCENARIO_LABELS = {
    "S1_non_cross_normal": "S1：同區／正常",
    "S2_cross_normal": "S2：跨區／正常",
    "S3_non_cross_ftp": "S3：同區／FTP burst",
    "S4_cross_ftp": "S4：跨區／FTP burst",
}


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def analyze(root: Path):
    if not (root / "FORMAL_COMPLETE.ok").is_file():
        raise RuntimeError("Formal evaluation is not marked complete")
    audit = json.loads((root / "packet_audit_FORMAL.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PACKET_AUDIT_PASSED" or audit.get("scenarios") != 80:
        raise RuntimeError("All 80 formal packet histories must pass audit")

    source = _read_csv(root / "summary_FORMAL.csv")
    if len(source) != 80:
        raise RuntimeError(f"Expected 80 formal runs, found {len(source)}")
    scenarios = tuple(row[0] for row in SCENARIOS)
    expected = {(method, scenario, seed) for method in METHODS
                for scenario in scenarios for seed in EXPECTED_PAIRS}
    observed = {}
    manifests = {}
    for seed, traffic_seed in PAIRS:
        path = root / f"checkpoint{seed}_traffic{traffic_seed}" / "trace_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest["checkpoint_seed"] != seed or
                manifest["traffic_seed"] != traffic_seed):
            raise RuntimeError(f"Manifest seed mismatch: {path}")
        manifests[seed] = manifest

    for row in source:
        method, scenario = row["variant"], row["scenario"]
        seed, traffic_seed = int(row["checkpoint_seed"]), int(row["traffic_seed"])
        key = (method, scenario, seed)
        if key not in expected or key in observed:
            raise RuntimeError(f"Missing, unexpected or duplicate formal run: {key}")
        if (row["status"] != "FORMAL_COMPLETE" or
                traffic_seed != EXPECTED_PAIRS[seed] or
                row["trace_sha256"] != manifests[seed]["trace_sha256"] or
                float(row["background_load_mbps"]) != 240.0):
            raise RuntimeError(f"Invalid seed, trace or status: {key}")
        if int(row["generated_packets"]) != (int(row["delivered_packets"]) +
                                              int(row["dropped_packets"]) +
                                              int(row["in_flight_at_horizon"])):
            raise RuntimeError(f"Packet conservation failed: {key}")
        for metric, _, _, _ in METRICS:
            if not math.isfinite(float(row[metric])):
                raise RuntimeError(f"Nonfinite value: {key}/{metric}")
        observed[key] = row
    if set(observed) != expected:
        raise RuntimeError(f"Formal result coverage mismatch: missing {len(expected - set(observed))}")
    for seed in EXPECTED_PAIRS:
        for scenario in scenarios:
            counts = {int(observed[(method, scenario, seed)]["generated_packets"])
                      for method in METHODS}
            if len(counts) != 1:
                raise RuntimeError(f"Unpaired packet injection: {seed}/{scenario}")

    previous = {(row["scenario"], row["method"]): row
                for row in _read_csv(root / "four_metrics_mean_sd.csv")}
    if len(previous) != 16:
        raise RuntimeError("Expected 16 existing mean/SD summaries")
    results = []
    for scenario in scenarios:
        for method in METHODS:
            baseline = previous[(scenario, method)]
            if int(baseline["n_seeds"]) != 5:
                raise RuntimeError(f"Existing summary has unexpected n: {scenario}/{method}")
            for metric, _, label_zh, scale in METRICS:
                sample = [float(observed[(method, scenario, seed)][metric]) * scale
                          for seed in EXPECTED_PAIRS]
                mean = statistics.mean(sample)
                sd = statistics.stdev(sample)
                if (not math.isclose(mean, float(baseline[f"{metric}_mean"]),
                                     rel_tol=1e-12, abs_tol=1e-10) or
                        not math.isclose(sd, float(baseline[f"{metric}_sd"]),
                                         rel_tol=1e-12, abs_tol=1e-10)):
                    raise RuntimeError(f"Mean/SD mismatch: {scenario}/{method}/{metric}")
                half_width = T_CRITICAL_95_DF4 * sd / math.sqrt(5)
                results.append({
                    "scenario": scenario,
                    "method": method,
                    "metric": metric,
                    "metric_label_zh": label_zh,
                    "n_seeds": 5,
                    "mean": mean,
                    "sample_sd": sd,
                    "ci95_lower": mean - half_width,
                    "ci95_upper": mean + half_width,
                    "ci95_half_width": half_width,
                })
    if len(results) != 64:
        raise RuntimeError(f"Expected 64 intervals, found {len(results)}")
    return results


def write_outputs(output: Path, results):
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "four_metrics_95ci.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    lookup = {(row["scenario"], row["method"], row["metric"]): row
              for row in results}
    lines = [
        "# 240 Mbps 四方法主結果：四場景、四項指標的 95% 信賴區間", "",
        "以五組凍結 checkpoint seeds 11／22／33／44／55，分別配對未參與訓練的測試流量 seeds 101／202／303／404／505。",
        "四方法均關閉探索（epsilon＝0），在相同 240 Mbps 全網背景負載與相同場景下評估。每個方法／場景的統計樣本數 n＝5；80 場不是 80 個獨立 seeds。", "",
        "每個 seed 先得到一個場景指標，再對五個值計算平均值（mean）、樣本標準差（sample standard deviation）與未校正的雙側 95% 信賴區間（95% confidence interval, CI）。",
        "公式：95% CI = mean ± t(0.975, 4) × SD / √5；t(0.975, 4) = 2.776445。這裡估計的是各方法／場景的平均指標，不是跨方法差值或顯著性檢定。原有誤差棒（error bars）表示 ±1 SD，**不是** 95% CI。", "",
    ]
    for metric, _, label_zh, _ in METRICS:
        lines += [f"## {label_zh}", "",
                  "| 場景 | 方法 | 平均值 ± SD | 95% CI |",
                  "|---|---|---:|---:|"]
        for scenario, _, _ in SCENARIOS:
            for method in METHODS:
                row = lookup[(scenario, method, metric)]
                lines.append(
                    f"| {SCENARIO_LABELS[scenario]} | {LABELS[method]} | "
                    f"{row['mean']:.2f} ± {row['sample_sd']:.2f} | "
                    f"[{row['ci95_lower']:.2f}, {row['ci95_upper']:.2f}] |"
                )
        lines.append("")
    lines += [
        "## 解讀限制", "",
        "- 每組僅五個獨立 seed 對；Student t 區間假設跨 seed 觀測近似獨立，且母體分布可用常態模型近似。樣本數太少，無法可靠驗證常態性。",
        "- 64 個逐項區間未作多重比較（multiple comparisons）校正；不同方法的兩個 CI 是否重疊，不能替代配對方法差值的統計檢定。",
        "- 封包損失率以百分比表示，CI 端點單位也是 %；端到端延遲只對已交付封包計算，須與損失率一起閱讀。",
        "- 結果限於目前 240 Mbps、既定星座／服務／流量場景；四方法末段學習曲線不足以證明全部充分收斂，故僅稱等互動預算比較（fixed-interaction-budget comparison）。", "",
        "資料來源：上層目錄的 `summary_FORMAL.csv`（80 場），並逐項核對 `four_metrics_mean_sd.csv`；逐封包稽核 `packet_audit_FORMAL.json` 已通過。", "",
    ]
    md_path = output / "FOUR_METRICS_95CI_zh.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_dir.resolve()
    rows = analyze(root)
    write_outputs(args.output_dir.resolve(), rows)
