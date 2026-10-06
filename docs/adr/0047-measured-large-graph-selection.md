# ADR0047: measured canvas selection with stable metadata

Status: accepted on Mac arm64 (2026-10-06); hosted successor verification recorded separately in HANDOFF§44.

Selecting an outline node on a declared500-node native-valid chain took181ms
at p95 in the warm owned Chrome/dev fixture. Workbench rebuilt every canvas-card
data object on selection; clearing empty edge selection also rebuilt all wire
metadata. Rendering cost grew with unrelated graph nodes.

Separate native graph/config/report/layout/measured-size data from selected flags.
Unselected node/wire objects retain their identities. Selected overlays remain
explicit; graph, layout, report, measured geometry and expanded module changes
still invalidate metadata. Native operation contracts, execution, history, saved
documents, diagnostic provenance and serving identities are unchanged. There is
no custom equality comparator that could conceal a real native-data change.

Measure actual owned headless Chrome and native HTTP on declared100/500/1000node
metadata-only tensor-input/ReLU chains. Record environment/source/graph hashes and
all20validation/10load/20outline-selection observations after discarding one warmup
per operation/size. Nearest-rank p95 and frame-observer/debounce overhead are
declared. Every graph must pass native validation; exact saved graph/UI/hash and
disabledUndo are checked after navigation. No latency target is an assertion.

Matched before/after raw evidence and methodology are published in
benchmarks/results/editor_selection_{before,after}.json and
editor_selection_summary.md. At500nodes observed selection p95 is181→54ms;
at1000nodes285.4→149.8ms. The500-node load p95 changes746.2→847.4ms;
no load improvement is claimed. These are descriptive single-laptop/headless
Chrome/dev-mode/synthetic-chain measurements, with low sample counts and no CPU
isolation. They do not establish add/update/pan/zoom/frame-time, representative
architectures, production build, heap/RSS, native execution speed, GPU, broader
platforms, outside-user productivity or accessibility certification.

Functional acceptance independently requires the full native/live suites,
editor build/types/Node, actual curl/native hash matching, all existing editor
regressions and source-deleted native/JSON/cache/local-tracker recovery. No
existing test is weakened. HANDOFF§44 records actual failures, intermediate
measurements, final gate results and publication/hosted state.

Final Mac correctness gates pass:1239native tests/1skip/14deselected559.86s,
14actualOllama live tests33.84s,280module build/typecheck/33Node191.00025ms,
coverage/pin/source-hash/diff audit, literalcurl native hashes/typed ports at all
three sizes, all12original editor journeys and137file/11database physically
source-deleted native/JSON/cache/local-tracker recovery. One arrangement setup
race was repaired by waiting for the fresh native load receipt and comparing
exact native canonical seed before any action; original assertions remain intact.
