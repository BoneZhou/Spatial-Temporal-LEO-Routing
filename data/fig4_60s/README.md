# Fig. 4: 60-second S4 packet-loss reference data

This directory is the compact, public replotting snapshot for the paper's
60-second Fig. 4 packet-loss panel. It contains four methods and five paired
runs (20 runs total):

| Checkpoint seed | Traffic seed |
|---:|---:|
| 11 | 101 |
| 22 | 202 |
| 33 | 303 |
| 44 | 404 |
| 55 | 505 |

Every run uses S4 cross-region FTP traffic, 240-Mbps background offered load,
30 warm-up slots, 500 baseline slots, 250 FTP-burst slots, 2250 recovery/
follow-up slots and a fixed 3000-slot measured horizon. A slot is 20 ms, so
the measured horizon is 60 seconds; the 30 warm-up slots are excluded.

`slot_metrics_FORMAL.csv` stores per-slot measurements.
`birth_cohorts.csv` stores generated, eventual-dropped and still-pending counts
by packet birth slot. `trace_manifest.json` preserves the exact paired NHPP and
FTP injection trace. Full packet-path logs and checkpoints are intentionally
excluded from the compact package.

Fig. 4 loss is computed for each run as eventual dropped births divided by all
births in a trailing 10-slot (0.2-second) window. In-flight packets at the
horizon are censored rather than counted as drops. The plotted line is the
five-run mean and the shaded band is sample standard deviation (`ddof=1`).

Replot with:

    python reproduce.py burst-plot --source reference
