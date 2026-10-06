# ADR0065: bounded flat JSONL sources and native transport

Status: accepted; Mac native/live/editor/recovery verified, 2026-10-06.

HANDOFF§59 item4 adds `operations/jsonl_source.py`, registered separately from
existing `tabular_ops.py`. No new dependency; standard-library JSON parsing and
existing pandas DataFrame.from_records. Preserve tabular/core.py and production/
pipeline.py and all prior pinned execution modules.

Each nonblank LF-delimited UTF-8 line is a nonempty flat object. Columns follow
first occurrence; missing keys/null become native missing values, with pandas dtype
inference and no date conversion or nested flattening. Row IDs are zero-based
nonblank record positions. Blank ASCII space/tab/CR lines are ignored; Unicode
line separators inside strings remain content. Duplicate keys, malformed objects,
arrays/nesting, invalid UTF-8/surrogates, nonfinite values and integers beyond
+/-(2^53-1) refuse with physical-line E_JSONL_RECORD diagnostics. Deep parser
recursion also refuses. Bounds: 50,000,000 bytes, 1,000,000 bytes per record,
200,000 records, 256 columns, 2,000,000 cells; column names <=256 UTF-8 bytes.
Validate the full bounded file; source SHA-256 covers every original byte.

Static inference has a bounded eight-entry path/mtime/size cache; execution always
rereads and validates bytes. Source nodes bypass node cache; descendants use actual
source content. New module joins cache implementation identity conservatively,
invalidating existing cache entries, without changing fitted-serving source pins.
Worker source_recorded and connection export requirements include JSONL. Bundles
list required local files without embedding them. Packages embed original JSONL
bytes only after explicit opt-in, with verified SHA and `.jsonl` suffix. The legacy
includeCsv API field now covers CSV and JSONL; existing CSV behavior is unchanged,
while the UI accurately labels both. Packages remain inert and install nothing.

Authenticated separate loopback CPU workers accept CSV/JSONL source snapshots,
verify the exact source map/bytes/hash and materialize matching suffixes. Existing
8 MiB transport/package bounds and operation allowlist remain; predictions_export
is still outside the remote allowlist. No remote/cloud/GPU or arbitrary JSON claim.
Existing fitted-pipeline capture already accepts generic table sources: source
lineage and native learned artifacts persist, serving does not read source files.

SYNTHETIC jsonl_regression preserves the existing housing example's 412 records
and native transforms/OLS. New focused tests independently compare dataframe and
native learned predictions, cache changes, deleted source + backup/restore,
package/bundle provenance and an actual separate authenticated worker. Owned Chrome
journey inspects source records, cache, explicit package import, actual serving,
replay and in-sample labels. Recovery restores exact version/release/source/trace/
project and stable monitor fields (only moving window.until is excluded), then
continues serving. Browser recovery deletes its workbench; native tests separately
delete the external JSONL file. No semantic housing-quality or dataset-portability
claim. Full acceptance and retained failures will be recorded in HANDOFF§64.

The owned editor runner now handles a transient ConnectionResetError during its
existing bounded service-readiness poll. Hosted item3 failed before Chrome launch
at cache-seed startup; evidence and final hosted outcomes remain in HANDOFF§64.
JSONL source clicks wait for actual finished UI state after native completion.
No browser assertion or error check is removed.

Integrated JSONL/cache verification also corrects the cache-retention seed helper
to select its explicit project (`any_project=False`), retaining exact19/3/16
assertions when other projects have cached nodes. Native store/cache code is unchanged.

Final local acceptance: native1361 passed/1 skipped/25 deselected558.54s,1941
retained warnings; live25; all21 editor runners; integrated sealed/GC recovery240
files/11DBs/16 browser cases. No hosted acceptance implied; HANDOFF§64 records
actual CI status and retained failures.
