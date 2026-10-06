# ADR0067: explicit CUDA training device, verified on a real GPU

Status: accepted (2026-10-06); CPU paths on Mac arm64 and hosted CI, CUDA on the user's RTX 5060 Ti (HANDOFF§68).

Model-graph training ran only on CPU and had no device setting, so the project could
not claim any GPU execution. The user's GPU machine (Ubuntu 24.04 on WSL2, RTX 5060 Ti
16 GB, compute capability 12.0) is reachable over Tailscale.

Decision: an explicit `device` in the training run configuration (`cpu` default,
`cuda` opt-in), never chosen automatically.

- `device="cuda"` without a usable CUDA device fails the run with
  `E_DEVICE_UNAVAILABLE` before any data load or checkpoint.
- Parameters are initialized on CPU from the seed, then moved; each batch moves to
  the device; validation results return to CPU. Data order uses the same CPU
  generator, so seeds fix initialization and order on both devices.
- Checkpoints contain CPU tensors only (model and optimizer state), so a CUDA-trained
  checkpoint loads with `weights_only` on any machine and the existing CPU serving,
  inference and portable Keras/JAX adapters accept it unchanged.
- `run_started` records `hardware`: device, GPU name, capability, memory, CUDA runtime,
  cuDNN, and that CUDA kernels are not bitwise reproducible.
- The editor Train tab has a Device select.

Measured on the GPU machine (torch 2.10.0+cu128, Python 3.11 in the user's `main`
conda env, not the Mac's pinned torch 2.14.1 CPU): same-seed first-step loss
2.2873857 (CPU) vs 2.2873826 (CUDA), last step within 4e-6 on the 60-image fixture.
SYNTHETIC shapes, 2000 images, 8 epochs, batch 64, Adam: CPU 14.29 s wall, final
val acc 0.5525; CUDA 3.03 s, 0.5725 (`benchmarks/results/gpu_training_cpu_vs_cuda.json`;
one run each, includes data loading, descriptive only).

Not provided: GPU for tabular/agent/RL/domain/procedure runs, multi-GPU, mixed
precision, automatic device choice, Apple MPS, GPU serving, deterministic CUDA
kernels, or GPU coverage in hosted CI (Linux CI has no GPU; the CUDA tests are marked
`gpu` and deselected by default). To reach the GPU machine, `psycopg[binary]==3.3.6`
and `gymnasium==1.3.0` (project pins) were installed into its `main` env; torch and
numpy were unchanged.

Verification: Mac: CPU default recorded, CUDA refused in-process and through the API
worker process. GPU machine (`pytest -m gpu tests/test_training_device.py`): CUDA
training matches the CPU start, the CUDA checkpoint is CPU-only and loads strictly
into the lowered model, and an API-submitted worker process trains on CUDA with the
device recorded.
