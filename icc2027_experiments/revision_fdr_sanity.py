"""Frozen FDR low-load test paired with a shortest-hop no-learning control."""
import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path
import torch
from .packet_validation import load_frozen_policy
from .baseline_methods import _shortest_hops
from .robustness_pilot import PilotSettings, make_trace
from .robustness_formal import run_scenario, trace_digest

class ShortestHopControl:
    def route_packets(self, decisions, snapshot):
        result = {}
        for d in decisions:
            result[d.packet_id] = min(d.feasible_neighbors, key=lambda peer:
                (_shortest_hops(peer,d.destination,snapshot.neighbors) or
                 (0 if peer == d.destination else 1000), peer))
        return result
    def on_slot_end(self, events, before, after):
        pass

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    policy,simulation,metadata=load_frozen_policy("fdr_marl_adapted","cpu",11)
    (args.output_dir/"training_sanity.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    settings=PilotSettings(seed=606,background_load_mbps=120,warmup_slots=30,
        baseline_slots=500,burst_slots=250,recovery_slots=750,
        measurement_slots=1500,recovery_definition="teacher_20261023")
    trace=make_trace(settings,simulation)
    results=[]
    for method,router in (("fdr_marl_adapted",policy),("shortest_hop_control",ShortestHopControl())):
        print(f"Starting low-load sanity: {method}",flush=True)
        row=run_scenario("S1_non_cross_normal",False,False,method,router,11,settings,
            simulation,trace,trace_digest(trace),args.output_dir,
            checkpoint_override=Path(metadata["checkpoint"]) if method=="fdr_marl_adapted" else Path(__file__))
        results.append(row)
        print(json.dumps({k:row[k] for k in ("variant","throughput_mbps","packet_loss_rate","mean_hops","average_queue_packets")}),flush=True)
    report={"status":"SANITY_EVALUATED_NOT_FORMAL_COMPARISON", "settings":asdict(settings),
        "checkpoint_is_240mbps_trained":True,"results":results,
        "purpose":"Low-load test of the existing frozen policy, not a low-load retraining or original-paper reproduction"}
    (args.output_dir/"sanity_results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    del policy
    gc.collect()

if __name__=="__main__":
    main()
