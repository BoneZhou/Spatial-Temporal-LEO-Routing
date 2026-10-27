"""Replot audited aggregates without changing any measurements (no Torch dependency)."""
import argparse
import csv
import json
import math
from pathlib import Path
from .publication_style import (MAIN, ABLATION, SCENARIOS, SCENARIO_LABELS,
                                METRICS, configure, axes_style, save)

def draw(source, destination, methods):
    import matplotlib.pyplot as plt
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {(r["scenario"], r["method"]): r for r in rows}
    if len(rows) != 16 or len(indexed) != 16:
        raise ValueError("Expected 16 distinct method/scenario aggregates")
    destination.mkdir(parents=True, exist_ok=True)
    for metric, label, stem in METRICS:
        fig, ax = plt.subplots(figsize=(7.5, 3.5))
        maxima = []
        for i, (method, name, color, _, _, hatch) in enumerate(methods):
            data = [indexed[s, method] for s in SCENARIOS]
            if any(int(r["n_seeds"]) != 5 for r in data):
                raise ValueError("Publication error bars require five independent seed pairs")
            mean = [float(r[metric + "_mean"]) for r in data]
            sd = [float(r[metric + "_sd"]) for r in data]
            if any(not math.isfinite(v) or v < 0 for v in mean + sd):
                raise ValueError("Invalid aggregate")
            maxima += [a + b for a, b in zip(mean, sd)]
            ax.bar([x + (i - 1.5)*.19 for x in range(4)], mean, width=.18,
                   yerr=sd, color=color, edgecolor="black", linewidth=.5,
                   hatch=hatch, label=name,
                   error_kw=dict(ecolor="black", capsize=4, elinewidth=1.2, capthick=1.2))
        ax.set_xticks(range(4), SCENARIO_LABELS)
        ax.set_ylabel(label)
        upper = max(maxima)*1.15
        ax.set_ylim(0, max(1, upper))
        axes_style(ax)
        ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(.5, 1.02),
                  columnspacing=1, handlelength=1.5)
        save(fig, destination / stem)
        plt.close(fig)
    (destination / "CAPTIONS.md").write_text(
        "S1: local/normal; S2: cross/normal; S3: local/FTP burst; S4: cross/FTP burst.\n"
        "Bars: mean ± sample standard deviation over five independent paired runs (n=5).\n"
        "240 Mbps background; fixed interaction budget. External methods are adapted baselines.\n"
        "These figures retain the original evaluation timing; they are not the revised 10–15 s burst evaluation.\n",
        encoding="utf-8")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    configure()
    root = Path("outputs/packet_fair_formal_load240_fiveseed")
    ablation = Path("outputs/packet_ablation_formal_load240_fiveseed")
    for source, count in ((root, 80), (ablation, 60)):
        audit = json.loads((source / "packet_audit_FORMAL.json").read_text(encoding="utf-8"))
        if not (source / "FORMAL_COMPLETE.ok").is_file() or audit.get("status") != "PACKET_AUDIT_PASSED" or audit.get("scenarios") != count:
            raise ValueError(f"Missing audited formal batch: {source}")
    draw(root / "four_metrics_mean_sd.csv", args.output_dir / "main", MAIN)
    draw(ablation / "ablation_four_metrics_mean_sd.csv", args.output_dir / "ablation", ABLATION)

if __name__ == "__main__":
    main()
