# ADR0079: compressed table outputs

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded in HANDOFF§86.

The data-scale benchmark (HANDOFF§82) found that a tabular run stores every intermediate
table as CSV in the content store, about 5.5× the input CSV (1.15 GB for a 208 MB file):
the first practical limit for large runs.

Decision: table outputs (`node_output`, value kind `table`) larger than 64 KiB are stored
gzip-compressed (level 1, `mtime=0`), with `encoding: "gzip"` and `rawBytes` in the
artifact metadata.

- Deterministic: the same table gives the same bytes and hash, so the content store
  still deduplicates identical outputs across runs.
- Small tables stay plain CSV; existing artifacts without `encoding` are read as before.
- The table inspector decompresses when the metadata says so. Serving pipelines only
  reference node-output hashes (their reference tables are separate artifacts), and
  garbage collection finds references in other records, not inside tables, so neither
  changes.
- Level 1 was chosen by measurement: at 10M rows level 6 cut storage to 521 MB but
  doubled run time (38 → 79 s); level 1 stores 565 MB at 45 s.

Measured on the Mac (`benchmarks/results/tabular_scale_mac_compressed.json`): stored
size 2.6–3.0× the CSV (was 5.2–5.6×); run time +7% to +17% (10M rows 38.3 → 44.8 s);
accuracy unchanged.

Not provided: recompressing artifacts written before this change, columnar formats
(Parquet would need a new dependency), compressing non-table artifacts, exporters that
decompress for external trackers (they receive the stored bytes with the metadata).

Verification: `tests/test_tabular_compression.py` (20,000-row run: source table
compressed with raw size recorded and exact row ids after decompression; non-table
outputs plain; a second run gives identical hashes; the inspector pages the last rows
correctly), the existing tabular suites, and the benchmark.
