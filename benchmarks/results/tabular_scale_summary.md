# Tabular data-scale benchmark (2026-10-06)

`benchmarks/tabular_scale.py`: the `production_sensors` pipeline (CSV source → split → fit standardize → apply ×2 →
logistic regression → metrics) on SYNTHETIC CSVs with the fixture's columns and a noisy linear label rule, one fresh
process per size, one run per size (no repeats: these indicate scale, not precise rates).

### Mac arm64 (project .venv, Python 3.13)

| rows | CSV MB | validate s | run s | µs/row | peak RSS MiB | artifacts MB | stored/CSV | val. accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.2 | 1.16 | 0.08 | 8.3 | 439 | 1 | 5.3× | 0.92 |
| 100,000 | 2.1 | 1.18 | 0.42 | 4.2 | 497 | 11 | 5.2× | 0.9206 |
| 1,000,000 | 20.8 | 1.25 | 3.76 | 3.8 | 811 | 111 | 5.4× | 0.9222 |
| 3,000,000 | 62.3 | 1.20 | 11.22 | 3.7 | 1518 | 343 | 5.5× | 0.9222 |
| 10,000,000 | 207.8 | 1.26 | 38.28 | 3.8 | 3179 | 1154 | 5.6× | 0.9222 |

### gpu-box Linux/WSL2 x86-64 (conda env main, Python 3.11; not the pinned environment)

| rows | CSV MB | validate s | run s | µs/row | peak RSS MiB | artifacts MB | stored/CSV | val. accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.2 | 2.42 | 0.27 | 27.2 | 852 | 1 | 5.3× | 0.92 |
| 100,000 | 2.1 | 1.28 | 0.76 | 7.6 | 884 | 11 | 5.2× | 0.9206 |
| 1,000,000 | 20.8 | 1.43 | 5.58 | 5.6 | 1086 | 111 | 5.4× | 0.9222 |
| 3,000,000 | 62.3 | 1.30 | 16.56 | 5.5 | 1533 | 343 | 5.5× | 0.9222 |
| 10,000,000 | 207.8 | 1.49 | 56.04 | 5.6 | 2900 | 1154 | 5.6× | 0.9222 |

Findings:
- Execution time grows linearly (≈3.8 µs/row on the Mac, ≈5.6 µs/row on the Linux machine at 10M rows); 10 million
  rows complete in 38 s / 56 s with peak memory about 3 GiB. No size failed.
- Validation stays ≈1.2–1.5 s: above 50 MB the static source check reads a 20,000-row sample (`staticSourceExact`
  false), so validation of large files is a sampled check of the schema, by design.
- Storage amplification: every intermediate table is stored in the content store, ≈5.5× the CSV bytes
  (1.15 GB of artifacts for a 208 MB CSV). This, not time or memory, is the first practical limit for large
  tabular runs; retention/GC (ADR 0050) and cache retention are the existing controls. Not changed here.
- Accuracy equals the label rule's ceiling for a linear model on `signal` (1 − arctan(0.25)/π ≈ 0.922) from
  1M rows on — a sanity check that results are correct at scale, not a quality benchmark.
Not measured: data larger than memory, Postgres/JSONL sources at scale, concurrent runs, repeated trials.

## After ADR 0079 (table outputs over 64 KiB gzip level 1), Mac

| rows | run s | µs/row | peak RSS MiB | artifacts MB | stored/CSV |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.09 | 9.1 | 444 | 0.6 | 3.0× |
| 100,000 | 0.49 | 4.9 | 515 | 5.4 | 2.6× |
| 1,000,000 | 4.47 | 4.5 | 893 | 55.2 | 2.7× |
| 3,000,000 | 13.29 | 4.4 | 1466 | 168.6 | 2.7× |
| 10,000,000 | 44.77 | 4.5 | 3398 | 565.3 | 2.7× |

Storage roughly halves (5.5× → 2.7× the CSV) for about 7–17% more run time; level 6 was measured at 10M rows (521 MB, 78.7 s) and rejected as too slow.
