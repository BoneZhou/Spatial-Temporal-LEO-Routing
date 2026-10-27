"""Recompute teacher-defined recovery and honest birth-cohort loss from saved data.

No Torch dependency. Old timing stays visible; this script never relabels it as 10–15 s.
"""
import argparse
import csv
import gzip
import json
import math
import statistics
from pathlib import Path
from .publication_style import MAIN, configure, axes_style, line_style, save

def rolling(values, width):
    result = []
    for i in range(len(values)):
        window = values[max(0, i-width+1):i+1]
        finite = [x for x in window if math.isfinite(x)]
        result.append(statistics.mean(finite) if len(window) == width and len(finite) == width else math.nan)
    return result

def recovery(queues, start, end, *, baseline_slots=50, rolling_slots=25, stable_slots=25, dt=.02):
    if start < baseline_slots or end >= len(queues):
        raise ValueError("Insufficient pre-burst or post-burst observation")
    qpre = statistics.mean(queues[start-baseline_slots:start])
    threshold = 1.10*qpre
    smoothed = rolling(queues, rolling_slots)
    excursion = any(v > threshold for v in smoothed[start:] if math.isfinite(v))
    for i in range(end, len(queues)-stable_slots+1):
        if all(math.isfinite(x) and x <= threshold for x in smoothed[i:i+stable_slots]):
            return dict(q_pre=qpre, threshold=threshold, recovery_time_s=(i-end)*dt,
                        status="recovered", threshold_excursion=excursion,
                        observation_window_s=(len(queues)-end)*dt)
    return dict(q_pre=qpre, threshold=threshold, recovery_time_s=None,
                status="right_censored", threshold_excursion=excursion,
                observation_window_s=(len(queues)-end)*dt)

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def cohorts(directory, slots, warmup, cache):
    """Dropped / all measured births, with unresolved births separately censored."""
    if cache.exists():
        return read_csv(cache)
    # A compact, derived birth ledger accompanies the public plotting data.
    # Fresh evaluations continue to reconstruct it from the full packet log.
    compact = directory / "birth_cohorts.csv"
    if compact.is_file() and not (directory / "packets_FORMAL.csv.gz").is_file():
        rows = read_csv(compact)
        if [int(row["index"]) for row in rows] != list(range(slots)):
            raise ValueError("Invalid compact birth ledger")
        return rows
    generated, dropped, pending = [0]*slots, [0]*slots, [0]*slots
    with gzip.open(directory / "packets_FORMAL.csv.gz", "rt", newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        birth, outcome = header.index("birth_slot"), header.index("outcome")
        for row in reader:
            index = int(row[birth])-warmup
            if not 0 <= index < slots:
                continue
            generated[index] += 1
            dropped[index] += row[outcome] == "dropped"
            pending[index] += row[outcome] == "in_flight_at_horizon"
    rows = [dict(index=i, generated=generated[i], dropped=dropped[i], pending=pending[i])
            for i in range(slots)]
    write_csv(cache, rows)
    return rows

def process(root, output):
    import numpy as np
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "cohort_cache"
    cache_dir.mkdir(exist_ok=True)
    per_seed, injection_audit = [], []
    for scenario in ("S3_non_cross_ftp", "S4_cross_ftp"):
        series = {method[0]: [] for method in MAIN}
        reference_injections = {}
        timings = set()
        for method, *_ in MAIN:
            for seed, traffic in ((11,101), (22,202), (33,303), (44,404), (55,505)):
                directory = root / f"checkpoint{seed}_traffic{traffic}" / method / scenario
                summary = json.loads((directory / "summary_FORMAL.json").read_text(encoding="utf-8"))
                rows = read_csv(directory / "slot_metrics_FORMAL.csv")
                warmup = int(summary["warmup_slots"])
                start = int(summary["baseline_slots"])
                end = start + int(summary["burst_slots"])
                timings.add((start, end))
                if [int(r["slot"]) for r in rows] != list(range(warmup, warmup+len(rows))):
                    raise ValueError("Missing or duplicate slot samples")
                counts = [(int(r["generated_packets"]), int(r["ftp_injected_packets"])) for r in rows]
                if traffic in reference_injections and reference_injections[traffic] != counts:
                    raise ValueError("Methods do not share identical injection")
                reference_injections[traffic] = counts
                cohort = cohorts(directory, len(rows), warmup,
                    cache_dir / f"{scenario}_{method}_{seed}.csv")
                births = [int(r["generated"]) for r in cohort]
                if births != [x[0] for x in counts]:
                    raise ValueError("Birth-cohort packet ledger differs from slot arrivals")
                generated = [float(x[0]) for x in counts]
                ftp = [float(x[1]) for x in counts]
                drops = [float(r["dropped"]) for r in cohort]
                pending = [float(r["pending"]) for r in cohort]
                queues = [float(r["average_queue_packets"]) for r in rows]
                # Ratio of rolling sums, not average of noisy slot ratios.
                total = rolling(generated, 10)
                loss = [100*d/g if g > 0 else math.nan for d,g in zip(rolling(drops,10), total)]
                censored = [100*d/g if g > 0 else math.nan for d,g in zip(rolling(pending,10), total)]
                delivered = [float(r["delivered_packets"]) for r in rows]
                delay_sums = [float(r["mean_delivered_delay_ms"] or 0)*n for r,n in zip(rows,delivered)]
                delay = [a/b if b > 0 else math.nan for a,b in zip(rolling(delay_sums,10), rolling(delivered,10))]
                data = dict(queue=rolling(queues,10), loss=loss, delay=delay, censor=censored,
                            background=rolling([(g-f)*.6 for g,f in counts],10),
                            offered=rolling([g*.6 for g,f in counts],10))
                series[method].append(data)
                rec = recovery(queues, start, end)
                per_seed.append(dict(scenario=scenario, method=method, checkpoint_seed=seed,
                    traffic_seed=traffic, **rec,
                    peak_queue=max(v for v in data["queue"][start:] if math.isfinite(v)),
                    peak_loss_percent=max(v for v in loss[start:] if math.isfinite(v)),
                    peak_delay_ms=max(v for v in delay[start:] if math.isfinite(v))))
                last_ftp = max((i for i,v in enumerate(ftp) if v), default=-1)
                injection_audit.append(dict(scenario=scenario, method=method, seed=seed,
                    generated=sum(births), ftp_packets=int(sum(ftp)),
                    ftp_expected=summary["ftp_sessions"]*1334,
                    session_start_s=start*.02, session_end_s=end*.02,
                    injection_end_s=(last_ftp+1)*.02, duration_s=len(rows)*.02,
                    complete=sum(ftp)==summary["ftp_sessions"]*1334))
                print(f"Analyzed {scenario}/{method}/{seed}", flush=True)
        if len(timings) != 1:
            raise ValueError("Burst timings differ")
        start, end = next(iter(timings))
        length = min(len(run["queue"]) for runs in series.values() for run in runs)
        time = np.arange(length)*.02
        for metric, label in (("queue", "Average queue length (packets/sat.)"),
                              ("loss", "Packet loss rate (%)"),
                              ("delay", "End-to-end delay (ms)"),
                              ("censor", "Unresolved birth cohort (%)")):
            fig, ax = plt.subplots(figsize=(7.5,3.5))
            for method in MAIN:
                values = np.array([run[metric][:length] for run in series[method[0]]])
                mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
                ax.plot(time[::5], mean[::5], label=method[1], **line_style(method))
                ax.fill_between(time, mean-sd, mean+sd, color=method[2], alpha=.12, linewidth=0)
            decorate(ax, start, end, label)
            ax.legend(frameon=False,ncol=4,loc="lower center",bbox_to_anchor=(.5,1.02),
                      columnspacing=.9,handlelength=2.2,fontsize=8)
            save(fig, output/f"{scenario}_{metric}", eps=False)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.5,3.5))
        for metric, label, color, ls in (("background","Background NHPP","#1F77B4","--"),
                                       ("offered","Background + FTP injection","#D32F2F","-")):
            values=np.array([run[metric][:length] for run in series[MAIN[0][0]]])
            mean,sd=values.mean(axis=0),values.std(axis=0,ddof=1)
            ax.plot(time,mean,color=color,linestyle=ls,linewidth=2,label=label)
            ax.fill_between(time,mean-sd,mean+sd,color=color,alpha=.12,linewidth=0)
        decorate(ax,start,end,"Offered load (Mbps)")
        ax.legend(frameon=False,ncol=2,loc="lower center",bbox_to_anchor=(.5,1.02))
        save(fig,output/f"{scenario}_offered",eps=False)
        plt.close(fig)
    write_csv(output/"recovery_per_seed.csv",per_seed)
    write_csv(output/"injection_audit.csv",injection_audit)
    aggregate=[]
    for scenario in ("S3_non_cross_ftp","S4_cross_ftp"):
        for method,*_ in MAIN:
            runs=[r for r in per_seed if r["scenario"]==scenario and r["method"]==method]
            row=dict(scenario=scenario,method=method,n=5,recovered=sum(r["status"]=="recovered" for r in runs),
                     threshold_excursion_runs=sum(r["threshold_excursion"] for r in runs),
                     q_pre_mean=statistics.mean(r["q_pre"] for r in runs))
            for metric in ("peak_queue","peak_loss_percent","peak_delay_ms","recovery_time_s"):
                values=[r[metric] for r in runs if r[metric] is not None]
                # Never replace censored recovery by an observation-window maximum.
                row[metric+"_mean"]=statistics.mean(values) if len(values)==5 else "censored: see per-seed"
                row[metric+"_sd"]=statistics.stdev(values) if len(values)==5 else ""
            aggregate.append(row)
    write_csv(output/"recovery_mean_sd.csv",aggregate)
    (output/"DEFINITIONS.md").write_text(
        "# Burst diagnostic definitions\n\n"
        "Original timing is preserved. Vertical lines bound FTP SESSION ARRIVALS; file injection may continue afterwards. "
        "All times exclude warm-up and label slot starts; queue is the post-service sample for that bin.\n\n"
        "Each seed is smoothed over 10 slots (0.2 s), then five seed means and sample SD are computed. "
        "The first 9 bins have no full window. All curves are truncated to the shortest shared observation window (no interpolation). "
        "Lines display every fifth slot (0.1 s) after smoothing; markers every 10 displayed points (1 s). Bands retain full resolution. "
        "Raw slot/packet files remain unchanged.\n\n"
        "Loss: eventual dropped measured packets / all measured births in each rolling birth-cohort window. "
        "In-flight packets are censored, not dropped; a separate unresolved-cohort curve shows this limit. "
        "This loss curve uses follow-up information and is not an online alarm. "
        "Delay: delivered-packet-weighted mean by DELIVERY time, using the same 10-slot window.\n\n"
        "Recovery: Qpre from 50 slots before session-arrival start; trailing 25-slot queue mean <=1.10*Qpre "
        "for 25 consecutive slots. Report the first qualifying window START minus session-arrival end, "
        "not its later confirmation time or the FTP-tail end. Unrecovered runs are right-censored. "
        "Peaks: per-seed maximum of the 0.2 s curves from session-arrival start to that run's horizon. "
        "Mean ± SD of recovery is withheld if any seed is censored.\n\n"
        "A zero recovery with no threshold excursion is NOT evidence of superior resilience; a saturated pre-burst queue can mask congestion. "
        "The tables include Qpre and threshold-excursion flags for that reason.\n\n"
        "PDFs preserve transparency. No EPS time-series is exported because EPS cannot represent the SD-band alpha faithfully. "
        "Figures are diagnostic until the required timing and FDR sanity checks pass.\n",encoding="utf-8")

def decorate(ax,start,end,label):
    for x in (start*.02,end*.02):
        ax.axvline(x,color="#666666",linestyle="--",linewidth=1.2)
    ax.axvspan(start*.02,end*.02,color="#F2F2F2",alpha=.35)
    ax.set_xlabel("Time since measurement start (s)")
    ax.set_ylabel(label)
    ax.set_ylim(bottom=0)
    axes_style(ax)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    configure()
    process(args.source_dir,args.output_dir)

if __name__=="__main__":
    main()
