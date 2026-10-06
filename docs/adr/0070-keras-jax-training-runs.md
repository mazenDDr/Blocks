# ADR0070: Keras and JAX training runs

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded in HANDOFF§71.

Keras 3 and JAX ran only forward passes, gradients and single SGD steps for the
portable subset (ADR0028/0029); model-graph training runs, checkpoints and run
history existed only for PyTorch.

Decision: `backend: pytorch|keras|jax` in the training run configuration.

- Keras/JAX runs share validation, data loading, the split record and the data order
  with PyTorch runs. Parameters are initialised by the seeded PyTorch lowering and
  copied into the backend executable in graph layout, so all three backends start
  from identical weights.
- Training appends an int64 target input and a mean cross-entropy loss node (ids
  chosen not to collide) and calls the executable's plain SGD step (p ← p − lr·grad)
  per batch on the backend. Validation runs the backend's forward pass.
- Checkpoints convert the backend parameters back to the PyTorch state dict (CPU
  tensors, `backend` recorded), so inference, image serving and the Keras/JAX
  serving adapters (ADR0061) load them unchanged. Events are those of PyTorch runs;
  `run_started` records the backend and its library versions.
- Refused (E_BACKEND_OPTIMIZER, before any checkpoint): Adam, momentum and CUDA on
  Keras/JAX. Graphs outside the portable subset are refused by the existing
  compatibility report.
- The editor Train tab has a Backend select.

Measured (reference CNN, 60-image SYNTHETIC fixture, seed 5, lr 0.05, batch 16,
2 epochs): first-step loss PyTorch 2.3366787, Keras 2.3366785, JAX 2.3366787;
last-step loss 2.2932959 on all three.

Not provided: Adam/momentum/weight decay on Keras/JAX, GPU for those backends,
mixed precision, Keras/JAX-native checkpoint formats (SavedModel/Orbax), or training
for graphs outside the portable subset.

Verification: 5 pytest cases (Keras and JAX runs matching the PyTorch start and
loading strictly into the PyTorch model; Adam, momentum and CUDA refusals) and an
owned Chrome journey selecting JAX in the Train tab and completing a native run.
