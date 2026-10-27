"""Stream-validate every compressed formal packet history before publication."""

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path

from .robustness_aggregate import SEED_PAIRS, VARIANTS
from .robustness_pilot import SCENARIOS


def check_packet(row, summary, destinations, trace):
    source = int(row["source"])
    destination = int(row["destination"])
    birth = int(row["birth_slot"])
    hops = int(row["hops"])
    path = [int(node) for node in row["path"].split("-")]
    if source not in range(45) or destination != destinations[source]:
        raise RuntimeError("Unexpected source/destination mapping")
    if not summary["warmup_slots"] <= birth < trace["horizon"]:
        raise RuntimeError("Measured packet birth outside measurement window")
    if row["traffic_class"] not in ("nhpp", "ftp"):
        raise RuntimeError("Unknown traffic class")
    expected_phase = (
        "baseline" if birth < trace["burst_start"] else
        "burst" if birth < trace["burst_end"] else
        "ftp_tail" if birth < trace["recovery_start"] else
        "recovery"
    )
    if row["birth_phase"] != expected_phase:
        raise RuntimeError("Packet birth phase does not match slot")
    if len(path) != hops + 1 or path[0] != source:
        raise RuntimeError("Path and hop count differ")
    crossings = sum(left // 15 != right // 15 for left, right in zip(path, path[1:]))
    if crossings != int(row["inter_plane_crossings"]):
        raise RuntimeError("Inter-plane crossing count differs from path")
    outcome = row["outcome"]
    if outcome == "delivered":
        if path[-1] != destination or not row["end_slot"]:
            raise RuntimeError("Delivered packet lacks destination/end slot")
        expected_delay = (int(row["end_slot"]) - birth + 1) * 20.0
        if not math.isclose(float(row["end_to_end_delay_ms"]), expected_delay, abs_tol=1e-6):
            raise RuntimeError("End-to-end delay inconsistent with slot timestamps")
        if float(row["queue_wait_ms"]) > expected_delay:
            raise RuntimeError("Queue wait exceeds elapsed end-to-end time")
        if not math.isclose(
            expected_delay, hops * 20.0 + float(row["queue_wait_ms"]), abs_tol=1e-6
        ):
            raise RuntimeError("Slot delay does not decompose into hops plus queue wait")
    elif outcome == "dropped":
        if row["drop_reason"] not in ("queue", "ttl", "no_route") or not row["end_slot"]:
            raise RuntimeError("Dropped packet lacks valid reason/end slot")
    elif outcome == "in_flight_at_horizon":
        if row["end_slot"] or row["drop_reason"]:
            raise RuntimeError("In-flight packet incorrectly classified")
    else:
        raise RuntimeError(f"Unexpected packet outcome {outcome}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--scenarios", nargs="+", choices=[r[0] for r in SCENARIOS],
                        default=[r[0] for r in SCENARIOS])
    parser.add_argument("--checkpoint-seeds", nargs="+", type=int,
                        default=[pair[0] for pair in SEED_PAIRS])
    parser.add_argument("--traffic-seeds", nargs="+", type=int,
                        default=[pair[1] for pair in SEED_PAIRS])
    args = parser.parse_args()
    if len(args.checkpoint_seeds) != len(args.traffic_seeds):
        raise ValueError("Checkpoint and traffic seeds must be paired")
    root = args.output_dir.resolve()
    if not (root / "FORMAL_COMPLETE.ok").exists():
        raise RuntimeError("Formal evaluation is incomplete")
    total_rows = 0
    audited_scenarios = 0
    scenarios = [r for r in SCENARIOS if r[0] in args.scenarios]
    if len(scenarios) != len(args.scenarios) or len(set(args.variants)) != len(args.variants):
        raise ValueError("Repeated scenarios or variants")
    expected_scenarios = len(args.checkpoint_seeds) * len(args.variants) * len(scenarios)
    for checkpoint_seed, traffic_seed in zip(args.checkpoint_seeds, args.traffic_seeds):
        seed_root = root / f"checkpoint{checkpoint_seed}_traffic{traffic_seed}"
        trace = json.loads((seed_root / "trace_manifest.json").read_text(encoding="utf-8"))
        for variant in args.variants:
            for name, cross, use_ftp in scenarios:
                scenario_root = seed_root / variant / name
                summary = json.loads(
                    (scenario_root / "summary_FORMAL.json").read_text(encoding="utf-8")
                )
                destinations = trace[
                    "cross_region_destinations" if cross else "same_region_destinations"
                ]
                counts = Counter()
                seen_ids = set()
                with gzip.open(scenario_root / "packets_FORMAL.csv.gz", "rt",
                               newline="", encoding="utf-8") as stream:
                    for row in csv.DictReader(stream):
                        packet_id = int(row["packet_id"])
                        if packet_id in seen_ids:
                            raise RuntimeError(f"Duplicate packet ID in {scenario_root}")
                        seen_ids.add(packet_id)
                        check_packet(row, summary, destinations, trace)
                        counts[row["outcome"]] += 1
                        counts[f"class_{row['traffic_class']}"] += 1
                if sum(counts[outcome] for outcome in (
                    "delivered", "dropped", "in_flight_at_horizon"
                )) != summary["generated_packets"]:
                    raise RuntimeError(f"Generated count mismatch in {scenario_root}")
                for outcome, key in (
                    ("delivered", "delivered_packets"),
                    ("dropped", "dropped_packets"),
                    ("in_flight_at_horizon", "in_flight_at_horizon"),
                ):
                    if counts[outcome] != summary[key]:
                        raise RuntimeError(f"Outcome count mismatch in {scenario_root}")
                if counts["class_ftp"] != summary["ftp_injected_packets"]:
                    raise RuntimeError(f"FTP class count mismatch in {scenario_root}")
                if not use_ftp and counts["class_ftp"]:
                    raise RuntimeError(f"FTP traffic in normal scenario {scenario_root}")
                total_rows += summary["generated_packets"]
                audited_scenarios += 1
                print(f"Audited {audited_scenarios}/{expected_scenarios}: {scenario_root}", flush=True)
    result = {"status": "PACKET_AUDIT_PASSED", "scenarios": audited_scenarios,
              "packet_rows": total_rows}
    (root / "packet_audit_FORMAL.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(result)


if __name__ == "__main__":
    main()
