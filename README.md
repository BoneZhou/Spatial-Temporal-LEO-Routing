# Spatial-Temporal Learning-Based Distributed Routing for Dynamic LEO Satellite Networks

GAT-LSTM-DQN packet routing for a dynamic 45-satellite Walker-Delta LEO network.
Includes the proposed method, three adapted learning baselines, strict
packet-level ablations, paired evaluation, FTP burst/recovery and plotting.

The compact data reproduce the completed **240-Mbps, five-seed** experiments.
This package is for review before publication. See
[reproducibility scope](docs/REPRODUCIBILITY.md) for differences from the older
paper draft and the FDR-adapted limitation.

## Install

Use Python 3.10 or 3.11. Run commands from this directory.

    python -m venv .venv

Activate with .venv\Scripts\activate on Windows, or
source .venv/bin/activate on Linux/macOS, then:

    python -m pip install -r requirements.txt

PyTorch, NumPy and Matplotlib are sufficient; GAT uses native PyTorch without
PyTorch Geometric. CPU is supported. For GPU training, install a CUDA build of
PyTorch appropriate for your GPU and select --device cuda.
requirements-tested.txt records the directly tested package versions.

## Reproduce figures and tables from included data

    python reproduce.py tables --source reference
    python reproduce.py plot --source reference
    python reproduce.py burst-plot --source reference

Outputs:

- outputs/reference/main/: four metrics, sample SD, 95% t intervals and
  paired cross-region tables.
- outputs/reference/ablation/: full model versus GAT-only, LSTM-only and DQN.
- outputs/figures/reference/main/ and ablation/: individual metric plots and
  three-panel throughput/loss/queue figures, in PDF/EPS/PNG.
- outputs/figures/reference/burst/: 30-second S3/S4 curves and recovery tables.
  FTP session arrivals are at 10-15 seconds after warm-up.
- outputs/figures/reference/training/: recorded training rewards, shown in
  separate panels because the methods use different reward functions.

Plotting recomputes statistics from per-seed records and requires no weights.
Reference data contain summaries, slot measurements and derived birth-cohort
counts, rather than full packet event logs.
[Data provenance and definitions](data/README.md).

## Run checks

    python reproduce.py check
    python reproduce.py test
    python reproduce.py smoke

Smoke trains all seven methods/variants for one tiny episode, a small batch,
and update interval of one slot. It is only a functional check and refuses
to overwrite an existing outputs/smoke run.

## Train and evaluate from scratch

Inspect commands if desired:

    python reproduce.py train --group main --device cuda --dry-run
    python reproduce.py train --group ablation --device cuda --dry-run

Run the full protocol:

    python reproduce.py train --group main --device cuda
    python reproduce.py train --group ablation --device cuda
    python reproduce.py main --device cuda
    python reproduce.py ablation --device cuda
    python reproduce.py burst --device cuda
    python reproduce.py tables --source fresh
    python reproduce.py plot --source fresh
    python reproduce.py burst-plot --source fresh

Use --device cpu if needed. Training runs sequentially; --seeds 11 limits a
training invocation to one configured seed. Repeating a training command skips
complete runs and resumes incomplete runs at the last episode checkpoint.
Formal evaluation uses five paired checkpoint/traffic seeds, frozen weights,
and zero exploration. The evaluation commands run the full packet audit.
Main evaluation must finish before combining ablation results.

Full runs are expensive. GraphPR uses 45 independent agents; replay/optimizer
checkpoints can occupy several GB per run. Weights and large packet logs
are generated locally and excluded from this package.

## Parameters and code

- paper_config.json: seeds, training budget, evaluation windows, FTP reference
  settings and relative output locations.
- icc2027_experiments/config.py: actual simulation, Proposed learning and
  adapted-baseline learning parameters.
- models.py / proposed_packet.py: GAT, LSTM, DQN, replay and updates.
- baseline_methods.py: GraphPR-adapted, FDR-MARL-adapted,
  DQN-IR-adapted (fixed-link).
- baseline_train.py: common packet training entry point.
- run_packet_fair_formal.py / run_packet_ablation_formal.py: frozen evaluation.
- robustness_formal.py / robustness_pilot.py: tracked environment, destination
  pairing, NHPP/FTP traces and recovery.
- packet_*aggregate.py / packet_formal_ci.py / packet_cross_region_paired.py:
  evaluation statistics.
- revision_figures.py / paper_plots.py / revision_burst_analysis.py /
  publication_style.py: plotting.

run.py supplies shared routing helpers and retains the earlier focal-packet
runner needed by supplementary robustness code. reproduce.py uses the final
packet-level protocol.

[Baseline sources and adaptation ledger](docs/BASELINES.md) describe features,
rewards and physical assumptions changed from the source papers. These are
research adaptations, not the original authors' official implementations.

No publication license has been chosen yet. Confirm the intended code/data
license and manuscript version before making the repository public.
