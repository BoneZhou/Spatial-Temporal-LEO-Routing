# Compact reference data

These are measurements from completed experiments, packaged for replotting and
recomputing tables without model checkpoints. They are not newly simulated
results produced while preparing this package.

| Directory | Contents | Original relative source |
|---|---|---|
| main | 80 per-run summaries, paired trace manifests and original audit record | outputs/packet_fair_formal_load240_fiveseed |
| ablation | 60 reduced-model summaries and matching manifests | outputs/packet_ablation_formal_load240_fiveseed |
| fig4_60s | 20 paper Fig. 4 S4 summaries, paired traces, 3000-slot measurements and compact birth ledgers | outputs/teacher_revision_20261023/burst60s_gpu_parallel |
| burst | 40 earlier 30-s S3/S4 recovery-diagnostic summaries, slot measurements and compact birth ledgers | outputs/teacher_revision_20261023/burst30s_gpu_parallel |
| training | 35 episode CSVs: four main methods and three reduced variants, five seeds each | outputs/packet_fair_seed11_load240_budget10m; outputs/packet_fair_multiseed_load240_budget10m; outputs/packet_ablation_load240_budget10m |

manifest.json records each shipped file's SHA-256, source relative path and
source SHA-256. Only checkpoint path metadata were changed to relative paths;
measured numeric fields were preserved. A source audit JSON is a historical
audit record, not an audit rerun against the compact package.

## Metric definitions

- Throughput: delivered measured packets x 1500 x 8 / measurement duration.
- Loss: dropped measured packets / generated measured packets. Aggregated plots
  multiply this fraction by 100 for percent.
- Delay: mean birth-to-delivery slot sojourn of delivered measured packets,
  including their delivery slot, multiplied by 20 ms.
- Queue: per-slot mean across the 45 shared satellite queues, then across the
  measurement interval.
- Every measured packet is delivered, dropped, or in flight at the horizon.
  Drop reasons split into queue overflow, TTL exhaustion and no route.
- Error bars: sample standard deviation across the five independent seed pairs.
  Tables additionally provide unadjusted two-sided 95% Student-t intervals.

## Burst birth ledgers

birth_cohorts.csv is a derived per-birth-slot count table from the original
packet logs. It contains generated, eventual dropped and still-pending packet
counts. The plotting code checks its births against slot arrivals before use.
It allows honest rolling birth-cohort loss and unresolved-cohort curves
without distributing tens of millions of packet records.

For fresh experiments, revision_burst_analysis.py derives the same ledger
from packets_FORMAL.csv.gz. Reference ledgers cannot reconstruct individual
paths; use fresh full event logs for packet-path auditing.

For the 60-second Fig. 4, each per-seed loss series uses a trailing ten-slot
(0.2-s) birth window before computing the five-seed mean and sample SD. The
window numerator is eventual drops among packets born in the window; the
denominator is all births in the window. In-flight-at-horizon packets remain
in the denominator and are separately censored rather than counted as drops.

The retained 30-second diagnostic additionally plots queue, delivered-packet
delay and unresolved cohorts. Delivered-packet delay is weighted by deliveries.
Recovery uses the last 50 slots before burst arrivals as Qpre, a 25-slot
trailing queue mean, and 25 consecutive slots at or below 1.1 x Qpre.
Unrecovered runs are right-censored; they are not replaced by the maximum
observation time.
