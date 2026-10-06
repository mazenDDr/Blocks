# Editor selection measurements — 2026-10-06

Actual warm headless Chrome154.0.8037.98 on this Apple M4 Pro (12logical CPUs,
24GiB RAM), macOS arm64/Darwin27.0.0, Node25.9.0, Vite dev/React development
mode. Python/package versions, source hashes, fixture hashes and every timing
sample are in editor_selection_before.json and editor_selection_after.json.
The product baseline is25ec83f; after-source App.tsx hash is recorded separately
because measurements precede the change's commit. Timing scripts are versioned
at tools/editor_performance_benchmark.py and apps/editor/smoke/performance.mjs.

Declared SYNTHETIC metadata-only graphs contain one N×4 tensor input followed
by99/499/999native ReLU nodes, connected by tensor wires. Every native validation
succeeds; saved graph/UI/hash and disabledUndo are unchanged after navigation.
No model training, learned outputs or scientific accuracy are represented.

Each graph discards its first operation of each kind, then records20validation,
10load and20outline-selection observations. p95 uses nearest rank
sorted[ceil(.95*n)-1];10load samples therefore report their maximum at p95.
Validation is a real browser fetch through the Vite proxy, including HTTP/JSON,
without data scans/dry runs. Load begins at an existing project select change and
ends after the exact project/load receipt/current native identity/all node DOMs
and two animation frames are ready; native250ms validation debounce is included.
Selection uses the existing outline Inspect button and ends after the correct
native inspector ID/aria-pressed state and two frames; frame observers add latency.
No Puppeteer round-trip is inside an individual timed interval.

All numbers below are milliseconds, before / after:

| Nodes | Native HTTP validation p95 | Project load p95 | Outline selection p95 |
|---|---|---|---|
| 100 | 6.8 / 7.0 | 429.9 / 450.0 | 62.1 / 48.1 |
| 500 | 28.4 / 30.3 | 746.2 / 847.4 | 181.0 / 54.0 |
| 1000 | 176.3 / 174.3 | 1470.9 / 1452.6 | 285.4 / 149.8 |

The change separates stable native card/geometry and wire metadata from selection
flags. Previously every selection rebuilt all card data, and clearing empty edge
selection rebuilt all wire objects. Other unselected objects now retain identity;
new graph/config/layout/geometry/native reports still invalidate their data.

At500nodes the observed selection p95 changes181.0→54.0ms. Load changes
746.2→847.4ms; this run does not establish a load improvement. At1000nodes
selection remains149.8ms, so that larger fixture still exposes latency work.
The experiment does not establish the full VISION§18.3 routine-editing target:
add/update, pointer interaction and pan/zoom/frame-time were not measured.

One laptop, one browser process per before/after run, dev build, a homogeneous
chain and low sample counts cannot establish universal platform/architecture
performance. No CPU isolation, heap/RSS, accelerator, production build, cold-start,
outside-user usability or formal accessibility claim. Existing user services stayed
running; no paid resources were used. A card-only intermediate measured500-node
selection p95 at137.4ms, motivating the wire change; that result is retained in
HANDOFF§44 rather than combined with the final samples. Functional validation
and cleanup are required independently of latency; no performance assertion was
weakened to pass a budget. Reproduce with a fresh output directory:

```bash
.venv/bin/python tools/editor_performance_benchmark.py --output /private/tmp/void-editor-performance-stable-cards-wires
```

## Selection-independent side panels (ADR0048)

A CPU profile showed the remaining 1000-node cost was side panels that list every
node (arrangement/movement/copy checklists, keyboard/insertion dropdowns, inspector
source options, outline rows, block library) re-rendering on each selection. They
now reuse unchanged rows and option lists. Same benchmark and machine; raw samples
and source hashes of every changed component are in editor_selection_panels.json.

| Nodes | Native HTTP validation p95 | Project load p95 | Outline selection p95 |
|---|---|---|---|
| 100 | 6.2 | 465.0 | 33.0 |
| 500 | 25.5 | 721.1 | 34.2 |
| 1000 | 168.5 | 1562.2 | 47.1 |

Two other runs on the same code measured 1000-node selection p95 at 54.1 and
48.5ms. Load p95 at 1000 nodes ranged 1517.6–1668.2ms over the three runs, against
1452.6ms before, so no load change is claimed. Limits above still apply.

```bash
.venv/bin/python tools/editor_performance_benchmark.py --output /private/tmp/void-editor-performance-memo-final-2
```
