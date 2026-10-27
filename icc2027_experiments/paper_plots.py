"""Portable Matplotlib plots for the compact paper data."""
import csv
from pathlib import Path
import statistics

from .publication_style import MAIN, SCENARIOS, SCENARIO_LABELS, METRICS, axes_style, save


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def combined(source, output, methods):
    import matplotlib.pyplot as plt
    rows = {(row["scenario"], row["method"]): row for row in read_csv(source)}
    # Final three-panel figure: throughput, loss and queue; delay kept separately.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (metric, label, stem) in zip(axes, METRICS[:3]):
        for index, (method, name, color, _, _, hatch) in enumerate(methods):
            means = [float(rows[scenario, method][metric + "_mean"]) for scenario in SCENARIOS]
            sd = [float(rows[scenario, method][metric + "_sd"]) for scenario in SCENARIOS]
            ax.bar([x + (index - 1.5) * .19 for x in range(4)], means,
                   width=.18, yerr=sd, label=name, color=color, hatch=hatch,
                   edgecolor="black", linewidth=.5,
                   error_kw=dict(ecolor="black", capsize=3, elinewidth=1))
        ax.set_xticks(range(4), [value.replace("\n", " ") for value in SCENARIO_LABELS],
                      rotation=12, fontsize=9)
        ax.set_ylabel(label, fontsize=13)
        ax.set_ylim(0, max(1, float(ax.dataLim.ymax) * 1.15))
        axes_style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=12)
    fig.subplots_adjust(top=.82, bottom=.2, wspace=.4)
    output.mkdir(parents=True, exist_ok=True)
    save(fig, output / "combined_three_metrics")
    plt.close(fig)


def training_curves(source, output):
    """Raw rewards use different definitions; plot one panel per method."""
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, (method, name, color, *_) in zip(axes.flat, MAIN):
        for seed in (11, 22, 33, 44, 55):
            rows = read_csv(source / f"seed{seed}" / method / "training_metrics.csv")
            ax.plot([float(row["total_train_decisions"]) / 1e6 for row in rows],
                    [float(row["reward_mean"]) for row in rows],
                    color=color, alpha=.25, linewidth=.9, label=f"seed {seed}")
        ax.set_title(name)
        ax.set_xlabel("Training packet-hop decisions (millions)")
        ax.set_ylabel("Mean training reward")
        axes_style(ax)
    fig.suptitle("Fixed interaction budget; rewards are not comparable across methods")
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, output / "training_rewards")
    plt.close(fig)
