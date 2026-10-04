# ADR 0010: TensorFlow/Keras 3 and JAX backends, compatibility reports, native exports, coverage ledger

Status: accepted (Milestone 6a). Builds on ADR 0001 (graph format, PyTorch lowering) and ADR 0005 (sharing). Sources: VISION 14.1-14.3, 15.3-15.4, 18.4, 23 (Milestone 6), 24 (A06, A18, A19), 26.
Vision, NLP and speech domain workflows are Milestone 6b and are not part of this record.

## Environment facts (checked, not assumed)

On macOS arm64 / CPython 3.13.12, `pip install` resolved and installed CPU wheels for **tensorflow 2.21.0, keras 3.15.1, jax 0.11.2, jaxlib 0.11.2** (pinned in `python/requirements.txt` with their
dependencies). `backends.availability()` really imports and runs each library (a tiny convolution); a backend that cannot run is reported `unavailable` with the exact exception text, its
compatibility reports say so, `compile_graph` refuses with `E_BACKEND_UNAVAILABLE`, and its tests skip with that reason. On this machine all three backends are available. Keras runs on its
TensorFlow backend (`KERAS_BACKEND=tensorflow`, verified at import; any other Keras backend makes the adapter report unavailable). No GPU is used or tested.

## Decisions

1. **Graph-level backend, per-node verdicts.** `Graph.backend` is one of `pytorch | keras | jax`. The portable operations keep their historical ids (`pytorch.nn.conv2d`, `core.sub`, ...; renaming
   would break every saved project); whether a backend can run an operation is decided per node by `backends.compat_report`, never by the id prefix. `validate()` stays the single place that infers
   shapes, and adds the compatibility errors of the *selected* backend (stable `E_BACKEND_*` codes with node and path), so the editor, the API and the CLI see the same failure before anything runs (A18).
2. **JAX = plain `jax.numpy` / `jax.lax` over an explicit parameter pytree** `{node id: {name: array}}`, not Flax NNX. Reasons: JAX's contract is pure functions with explicit state and randomness (VISION 14.1);
   a pytree keeps parameter sharing, weight exchange and gradients visible and testable with no extra dependency or object-graph semantics; NNX's mutable-module convenience would hide exactly what the
   workbench must show. Consequence: no layers library; initialization and layout live in our code and are declared.
3. **Keras = Keras 3 layers + TensorFlow ops, eager by default, `tf.function` when compiled.** Layers own the variables (Conv2D, SeparableConv2D, Dense, MaxPooling2D, ...); gradients come from `tf.GradientTape`.
4. **Layout is explicit and never silent.** The graph is NCHW for every rank-4 tensor and OIHW for convolution weights. JAX consumes that directly (`lax.conv_general_dilated` dimension numbers): no conversions.
   Keras is NHWC: `keras_spec.analyze` plans every transpose per node (`to_native` before convolution/pooling, `to_graph` before Linear, reductions, flatten of spatial maps, and when reading outputs or captured
   activations), records each as a `layout` conversion in the node's report entry, and emits it in the exported code with a `layout:` marker. Flatten of an NHWC tensor with H = W = 1 needs no transpose
   (same element order) and says so; any other flatten transposes back, otherwise Linear inputs would be permuted. Weights cross backends only in graph layout (OIHW, `(out, in)`); Keras converts OIHW <-> HWIO and
   `(out, in)` <-> `(in, out)` at its boundary (`weight_layout` conversions).
5. **Padding semantics.** PyTorch explicit padding becomes `tf.pad` zeros + a `valid` convolution on Keras (conversion listed) and lax lo/hi pairs on JAX; `padding='same'` uses the PyTorch lo = floor split on JAX and
   Keras' `same` (identical for stride 1, tested with an even kernel). `padding_mode` reflect/replicate/circular: **unsupported on Keras** (`E_BACKEND_UNSUPPORTED_PADDING_MODE`), supported on JAX via `jnp.pad`.
   Dilation > 1 with stride > 1 is unsupported on Keras. Max-pool padding / dilation / ceil_mode are unsupported on Keras; on JAX padding and ceil_mode work via `lax.reduce_window`, but dilation is refused because
   JAX has no gradient rule for max reduce_window with window dilation (found by the conformance test, see `E_BACKEND_UNSUPPORTED_DILATION`). Adaptive average pooling is supported when the output size divides the input
   (global average pooling included); otherwise refused. Groups are supported on both (tested on CPU).
6. **Initialization differs and is declared.** PyTorch: Kaiming-uniform; Keras: Glorot-uniform kernels + zero biases; JAX: Uniform(+-1/sqrt(fan_in)) from an explicit key (same key-splitting in the runtime and the exported
   `init_params`, tested equal). Initial values are never claimed equal across backends; comparison uses copied weights. The declaration is in every compatibility report and in every exported file's header.
7. **dtype.** float32 everywhere; float64 on PyTorch and Keras; **JAX refuses float64** (it would silently compute in float32 unless a global flag is changed, which this adapter will not do);
   int64 labels become int32 on JAX, listed as a `dtype` conversion, with a range check at feed time.
8. **No silent substitution, no dropped settings.** An unsupported node (an operation outside the portable subset such as `tensor.*`, `diag.*`, `code.block`, or an unsupported setting) is `unsupported` with a code and reason;
   `compile_graph` / `export_code` raise `BackendError(E_BACKEND_INCOMPATIBLE)` carrying the report; `lower_graph` / `generate_pytorch` refuse a graph whose backend is not `pytorch` (`E_BACKEND_MISMATCH`); the API refuses worker
   training runs for non-PyTorch graphs (`backend_training_unsupported`). Switching backend only changes the `backend` field: settings are never rewritten (tested: hash and config unchanged).
9. **Parameter sharing is preserved or the switch is refused.** Shared call sites reuse one Keras layer object / one pytree entry; gradients accumulate across call sites; the report lists the sharing groups;
   exports show the call site `shares parameters with <owner>`. Tested against PyTorch gradients.
10. **Explicit backend-specific nodes** with no claimed equivalence: `keras.layers.separable_conv2d` (Keras SeparableConv2D semantics, Keras padding) and `jax.lax.cumsum`. They shape-infer everywhere (the editor still draws them)
    but are rejected on every other backend with `E_BACKEND_OP`. Each is tested against its own native layer/function.
11. **Native exports are generated text** (like the PyTorch one: deterministic, `# node:` comments, never executed by the app): a `Model` class with Keras layers and `set_graph_weights()`, and `init_params` / `apply` functions
    for JAX. Generation is pure Python and works without the framework installed. A test executes each export in a **separate Python process** with no project imports and compares outputs.
12. **Coverage ledger is derived.** `python -m backends.coverage --write` renders `docs/COVERAGE.md` from the registry, the adapters' supported sets, a single conformance-case table (the same table the tests parametrize
    over), the reference workloads and a static scan of `tests/`; `GET /api/coverage` serves the same data to the editor; a test fails when the committed file is stale. `ARCH_DEDICATED` (which ops have a dedicated schematic) is
    checked against the editor source by a test.
13. **Training stays PyTorch-only.** Keras and JAX executables provide forward, loss, gradients and one plain SGD step (the A06-style evidence). There is no worker training run, checkpoint or run history for them and every
    surface (API error, editor panel, `docs`) says so.

## Tolerances

Declared per backend/dtype in `backends/tolerances.py` after measuring the reference workloads (numbers in `docs/CAPABILITIES.md`); they bound floating-point reassociation, not bit equality.

## Consequences / limits

CPU only; the portable subset is small (14 operations); no Keras padding modes other than zeros; no cross-backend checkpoint conversion (weights move as NumPy arrays in graph layout, explicitly); no mixed-backend
pipelines; `tf.function` / `jit` are used by the executables (and the benchmark) for graphs without data-dependent control flow only; max-pool gradient ties may route differently across frameworks (not exercised).
