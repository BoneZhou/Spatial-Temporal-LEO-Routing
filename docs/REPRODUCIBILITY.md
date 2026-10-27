# Reproducibility scope and review notes

This replaces the earlier core-only source package. It retains completed
packet-level experiments and their measured data; no new formal performance
experiment was substituted for the existing results.

## Experiment mapping

| Result | Protocol | Command |
|---|---|---|
| Main four-method comparison, S1-S4 | 240 Mbps, five paired seeds; 80 runs | reproduce.py main; tables; plot |
| Strict GAT/LSTM ablation | Three reduced variants, 60 new runs; reuse 20 full-Proposed runs | reproduce.py train --group ablation; ablation; tables; plot |
| Local/cross FTP burst and recovery | 240 Mbps background, fixed 30-s window; 40 runs | reproduce.py burst; burst-plot |
| Cross-region paired effects | Cross minus local within each seed, then mean/SD/CI | reproduce.py tables |
| Training trajectories | Recorded episode metrics; fixed interaction budget | reproduce.py plot |
| Inference benchmark | Complete route_packets call, including history/masking/selection | python -m icc2027_experiments.benchmark_packet_inference --help |
| FDR diagnostic and shortest-path control | Frozen adapted policy and low-load control | python -m icc2027_experiments.revision_fdr_sanity --help |

Training seeds 11/22/33/44/55 pair with held-out traffic seeds
101/202/303/404/505. Formal evaluation disables exploration.

## Settings used by the completed experiment

Actual simulation: 45 satellites (3 planes x 15), altitude 570 km, inclination
70 degrees, 20-ms slots, 1500-byte packets, 300-Mbps fixed ISLs, one shared
100-packet queue per satellite, 36 packet services per satellite per slot,
and 30-hop TTL. Inter-plane links require LOS and a 60-degree latitude cutoff.

Learning settings are in config.py; architectures and rewards are described
in BASELINES.md. LearningConfig retains the earlier 1300-episode default for
the focal-packet helper. The final packet protocol instead uses a safety
ceiling of 1000 episodes and an absolute ceiling of 10,000,000 packet-hop
transitions. It stops before an episode would exceed the budget; warm-up
decisions do not count. Epsilon decays by decisions and reaches its floor
at 8,000,000. Each training episode has 30 warm-up and 100 training slots.
GraphPR updates every 10 slots; the other methods every slot.

paper_config.json controls the reproduction runner. Its FTP entries are
checked against actual trace-generator defaults in robustness_pilot.py.
Physical and learning changes belong in config.py. The frozen-policy loaders
check simulation signatures; their relative checkpoint directory convention
is preserved.

Original aggregate evaluation uses baseline/burst/recovery lengths of
250/250/500 slots plus a seed-specific FTP completion tail. Revised burst
evaluation uses a fixed 1500-slot window, with FTP session arrivals at
10-15 seconds after warm-up. These are different timing/recovery protocols.

## Differences from the locally available earlier PDF

The available earlier draft reports queue size 640, 1300 training episodes,
load-sweep plots at 120/180/240 Mbps, and a Proposed inference time of 2.70 ms.
The completed packet-level experiment uses the settings above and four
scenarios at 240 Mbps; it uses different checkpoints and evaluation accounting.

This package reproduces completed main/ablation/burst data. It does not assert
that it reproduces every number or figure in that older PDF. No validated
source was identified connecting that PDF's load-sweep and 2.70-ms table
directly to the final packet protocol. The measured inference benchmark is
included; its results depend on hardware and timing scope.

Dijkstra/shortest-path control remains in the simulator and FDR diagnostic.
The completed five-seed main comparison contains four learning methods; it
does not include an audited Dijkstra load-sweep result.

## Interpretation limits

- FDR-MARL-adapted retains the reward used by its saved checkpoints. Its
  low-load behavioral sanity check failed. The adaptation ledger identifies
  its reward difference from the source paper; retain this limitation when
  interpreting results.
- Equal interaction budgets do not establish convergence. Raw rewards cannot
  be ranked across methods with different reward definitions.
- Figure error bars show sample SD over five pairs. Separate t intervals use
  df=4 without multiple-comparison correction.
- Delay conditions on successful delivery. In-flight packets are censored
  separately rather than counted as losses.
- Shipped audit JSON files record the original completed packet audit.
  Compact reference data support summary/cohort consistency checks, not
  rerunning every original packet-path audit. Fresh evaluation writes full
  logs and reruns those audits.
- Full optimizer/replay checkpoints are excluded. Replotting works from
  bundled data; new model evaluation requires training first.
- Resume signatures include source hashes. Baseline learning constants were
  centralized without changing their values, so original checkpoints with
  older source hashes must not be resumed under this release. Frozen
  inference still checks method, seed and physical settings.
- PyTorch/device versions can change floating-point training results.
  Record the environment and compare new results statistically.

## Review before publication

Confirm the manuscript version and its parameter/result mapping, resolve
FDR-adapted wording, and choose a code/data license. The prepared package
excludes private machine paths, manuscripts, local runtimes, test outputs
and full checkpoints.
