# Spatial-Temporal Learning-Based Distributed Routing for Dynamic LEO Satellite Networks

This repository provides the implementation, experiment configurations, compact reference data, evaluation scripts, ablation studies, robustness analysis, and plotting utilities for a spatial-temporal learning-based distributed routing framework for dynamic low Earth orbit (LEO) satellite networks.

The proposed method combines graph attention, long short-term memory, and deep Q-learning (GAT-LSTM-DQN) for packet routing in a dynamic 45-satellite Walker-Delta LEO constellation. The repository also includes three adapted learning baselines, strict packet-level ablations, paired evaluation, FTP burst/recovery experiments, and publication-quality plotting utilities.

The included compact reference data reproduce the completed **240-Mbps, five-seed** experiments. Large training checkpoints and full packet-event logs are intentionally excluded because of their storage cost. See [Reproducibility Scope](docs/REPRODUCIBILITY.md) for the exact scope of the released artifacts, differences from earlier manuscript versions, and the documented FDR-adapted limitation.

## Quick Start

Use Python 3.10 or 3.11 and run commands from the repository root.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

PyTorch, NumPy, and Matplotlib are sufficient. The graph-attention implementation uses native PyTorch and does not require PyTorch Geometric.

CPU execution is supported. For GPU training, install a CUDA-compatible PyTorch build and use:

```text
--device cuda
```

`requirements-tested.txt` records the package versions used for direct testing.

## Reproduce Figures and Tables from Included Reference Data

The included compact reference data can be used to regenerate the reported tables and figures without retraining the models.

```bash
python reproduce.py tables --source reference
python reproduce.py plot --source reference
python reproduce.py burst-plot --source reference
```

Generated outputs are written to:

- `outputs/reference/main/`  
  Main evaluation metrics, sample standard deviations, 95% t confidence intervals, and paired cross-region tables.

- `outputs/reference/ablation/`  
  Full-model results and packet-level ablations for GAT-only, LSTM-only, and DQN variants.

- `outputs/figures/reference/main/` and `outputs/figures/reference/ablation/`  
  Individual metric plots and three-panel throughput/loss/queue figures in PDF, EPS, and PNG formats.

- `outputs/figures/reference/burst/`  
  30-second S3/S4 burst curves and recovery tables. FTP-session arrivals occur 10-15 seconds after warm-up.

- `outputs/figures/reference/training/`  
  Recorded training-reward curves. These are shown in separate panels because the compared methods use different reward functions.

Plotting recomputes the reported statistics from per-seed records and does not require model weights.

The released reference data contain summary statistics, slot-level measurements, and derived birth-cohort counts rather than complete packet-event logs. See [Data Provenance and Definitions](data/README.md).

## Validation and Functional Checks

Run the repository consistency checks with:

```bash
python reproduce.py check
python reproduce.py test
python reproduce.py smoke
```

The smoke test trains all seven methods/variants for one small episode using a reduced batch size and an update interval of one slot. It is intended only as a functional check and refuses to overwrite an existing `outputs/smoke` run.

## Train and Evaluate from Scratch

The full training/evaluation commands can be inspected before execution:

```bash
python reproduce.py train --group main --device cuda --dry-run
python reproduce.py train --group ablation --device cuda --dry-run
```

Run the full protocol with:

```bash
python reproduce.py train --group main --device cuda
python reproduce.py train --group ablation --device cuda

python reproduce.py main --device cuda
python reproduce.py ablation --device cuda
python reproduce.py burst --device cuda

python reproduce.py tables --source fresh
python reproduce.py plot --source fresh
python reproduce.py burst-plot --source fresh
```

Use `--device cpu` when GPU acceleration is unavailable.

Training runs sequentially. The option

```text
--seeds 11
```

limits a training invocation to one configured seed. Repeating a training command skips completed runs and resumes incomplete runs from the most recent episode checkpoint.

Formal evaluation uses five paired checkpoint/traffic seeds, frozen model weights, and zero exploration. The evaluation commands perform the complete packet audit. Main evaluation must finish before the ablation outputs are combined.

Full training is computationally expensive. GraphPR uses 45 independent agents, and replay/optimizer checkpoints can require several gigabytes per run. Large model weights and packet-event logs are generated locally and are not included in this repository.

## Experiment and Implementation Files

The primary configuration and implementation files are:

- `paper_config.json`  
  Seeds, training budget, evaluation windows, FTP reference settings, and relative output locations.

- `icc2027_experiments/config.py`  
  Simulation parameters, proposed-method learning parameters, and adapted-baseline learning parameters.

- `models.py` and `proposed_packet.py`  
  GAT, LSTM, DQN, replay-buffer, and learning-update implementations.

- `baseline_methods.py`  
  GraphPR-adapted, FDR-MARL-adapted, and DQN-IR-adapted fixed-link baselines.

- `baseline_train.py`  
  Common packet-level training entry point for the adapted baselines.

- `run_packet_fair_formal.py` and `run_packet_ablation_formal.py`  
  Frozen-weight formal evaluation.

- `robustness_formal.py` and `robustness_pilot.py`  
  Tracked-environment evaluation, destination pairing, NHPP/FTP traffic traces, and recovery experiments.

- `packet_*aggregate.py`, `packet_formal_ci.py`, and `packet_cross_region_paired.py`  
  Evaluation aggregation and statistical analysis.

- `revision_figures.py`, `paper_plots.py`, `revision_burst_analysis.py`, and `publication_style.py`  
  Publication-quality plotting and figure generation.

`run.py` provides shared routing utilities and retains the earlier focal-packet runner required by supplementary robustness code. The final packet-level protocol is executed through `reproduce.py`.

## Methods and Baseline Adaptations

The proposed method uses spatial-temporal learning to support distributed packet routing in a time-varying LEO topology.

The repository also includes three learning-based baseline adaptations:

- GraphPR-adapted
- FDR-MARL-adapted
- DQN-IR-adapted

The adapted baselines are research implementations aligned to the common packet-level evaluation framework. They are **not** official implementations released by the original authors.

See [Baseline Sources and Adaptation Ledger](docs/BASELINES.md) for the source references and for the features, rewards, and physical assumptions modified during adaptation.

## Evaluation Protocol

The formal evaluation protocol uses:

- five paired checkpoint/traffic seeds,
- frozen model weights,
- zero exploration,
- packet-level auditing,
- paired cross-region comparisons,
- sample standard deviations,
- 95% t confidence intervals,
- packet-level ablations, and
- FTP burst/recovery analysis.

The paired design is used to reduce evaluation variance when comparing routing methods under matched traffic and network conditions.

## Reproducibility Scope

This repository supports two reproducibility paths:

1. **Reference-data reproduction**  
   The included compact per-seed records can regenerate the reported tables, confidence intervals, and publication figures without model retraining.

2. **Full training and evaluation**  
   The released source code and configuration files can be used to retrain the methods and rerun the formal packet-level evaluation protocol.

Large replay/optimizer checkpoints, trained weights, and complete packet-event logs are intentionally excluded because of storage cost. These artifacts are generated locally during full runs.

For the exact released-data scope, differences from older manuscript versions, and known implementation limitations, see:

[Reproducibility Scope](docs/REPRODUCIBILITY.md)

## Data Provenance

The compact reference data contain summaries, slot measurements, and derived birth-cohort counts rather than full packet-level event logs.

Definitions, provenance, and interpretation of the released data are documented in:

[Data Provenance and Definitions](data/README.md)

## Citation

If you use this repository in academic work, please cite the associated manuscript. Formal bibliographic information will be added after publication.

```bibtex
@inproceedings{spatiotemporalLEOrouting2027,
  title  = {Spatial-Temporal Learning-Based Distributed Routing for Dynamic LEO Satellite Networks},
  note   = {Manuscript},
  year   = {2027}
}
```

## License

No publication license has been selected yet. Confirm the intended code/data license and manuscript version before making the repository public.

## Contact

For implementation or reproducibility questions, please open an issue in this repository.
