# ADR 0001: Local execution path and authoritative graph format

Status: accepted (Phase 0). Sources: VISION sections 15, 16, 17, 26; PLAN.md.

## Decisions

1. **The JSON graph is the single source of truth.** It is a Pydantic v2 model
   (`python/graph_core/schema.py`) shaped like VISION 16.2: `schemaVersion`, `graphKind`, `backend`,
   `nodes[{id,type,version,config,stateRef}]`, `edges[{id,kind,from{node,port},to{node,port}}]`.
   Models allow extra fields, so unknown operations, configs and future fields survive load/save.
2. **Layout is a separate document** (`*.ui.json`, `UiDoc.positions`). The semantic hash is
   SHA-256 over canonical JSON (sorted keys, nodes and edges sorted by id) of the spec only, so moving
   nodes or reordering declarations never changes it. Raw configs are hashed, not defaults-filled, so a
   future default change cannot silently alias two graphs.
3. **Operations are registered objects** (`graph_core.registry.Operation`) with typed ports, a
   pydantic config (defaults, `extra=forbid`), `infer_shape`, `param_count`, `lower`, `explain` and
   `codegen`. `in_channels` / `in_features` are `"infer"` or a locked int; inference resolves the
   placeholder, a locked value that disagrees yields `E_CHANNEL_MISMATCH` / `E_FEATURE_MISMATCH`.
4. **Shapes are symbolic only in the batch dim** (`"N"`, index 0). Everything else is an int.
5. **Validation never raises on bad input**; it returns diagnostics
   `{code, severity, nodeId, port, path, message, fixes[]}`. Unknown ops are diagnostics
   (`E_UNKNOWN_OP`) that block execution but not loading. Nodes downstream of a failure are not
   re-reported.
6. **Execution = lowering to `torch.nn.Module`** (`GraphModule`): one submodule per node, named by node
   id, executed in a deterministic topological order. No `eval`/`exec`. Activation capture is opt-in
   via forward hooks (`capture_activations`) and does not change outputs.
7. **Code export** is generated text only (readable PyTorch, `# node: <id>` comments as the source
   map). It is never executed by this project; tests parse it with `ast` and compare structure.
8. **Worker**: one OS process per run (`multiprocessing` spawn) running a fixed procedure
   (seeded, CrossEntropy, SGD or Adam). Cooperative cancel is checked before every batch.
9. **Persistence**: content-addressed files in `.workbench/artifacts/<sha256>` (atomic temp+rename) and
   SQLite `.workbench/meta.db` (WAL) for runs, events and artifact records. Run states follow VISION 17.1
   (subset: queued, preparing, running, cancelling, completed, failed, cancelled); illegal transitions raise.
10. **Checkpoints** hold model, optimizer, torch CPU RNG state, step, epoch, graph hash and a
    `complete`/`partial` status. A cancelled run writes a `partial` checkpoint at the batch boundary.

## Deviation from PLAN.md

PLAN says worker events travel "over a queue then persisted". Instead the worker appends events and
artifacts directly to the SQLite/CAS store. Reason: no unbounded-queue or lost-event hazard, the run
continues and stays inspectable if the submitting process disappears (A10), and Phase 1's SSE will read
the same table with `seq` as the resume cursor. The submitting process only creates the run row,
spawns the worker and signals cancellation via a `multiprocessing.Event`.

## Consequences / limits

- Only the `pytorch` backend, graph kind `model`, float32 image training are supported.
- Exact resume from a checkpoint is not implemented; checkpoints record what it would need.
- Broadcasting is not implemented; binary ops require equal shapes.
