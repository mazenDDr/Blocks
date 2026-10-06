# ADR0073: data-parallel training across worker processes

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded in HANDOFF§75.

Model-graph training used one process. Distributed training was listed as missing.

Decision: `workers: 1–8` in the training run configuration (PyTorch backend, CPU).

- The run's worker process is rank 0 and spawns `workers − 1` helpers that join a gloo
  process group on loopback (`worker/distributed.py`). Every rank lowers the same
  graph and loads the same image folder; rank 0 broadcasts its seeded initial
  parameters, and every rank derives the same data order from the seed.
- Each global batch is split across ranks (`tensor_split`, so uneven final batches
  work). Each rank backpropagates the SUM of cross-entropy over its shard; gradients
  are summed with all_reduce and divided by the global batch size; every rank applies
  the same optimizer step (SGD with momentum or Adam), so replicas stay identical.
  This equals single-process training on the same batches up to float summation order.
- Rank 0 alone evaluates and records events and checkpoints (unchanged format);
  `run_started` records `workers` and the parallelism. Helpers stop on run end,
  cancellation or failure.
- Refused (E_DISTRIBUTED_CONFIG): workers > 1 with Keras/JAX or CUDA.
- Editor Train tab: Workers field.

Measured against single-process training (reference CNN, 60-image SYNTHETIC fixture,
batch 10 with an uneven last batch, 2 epochs): 2 workers SGD max loss difference
2.9e-7, max parameter difference 6.3e-8; 3 workers Adam 3.8e-7 and 3.2e-5 (Adam's
per-parameter normalisation amplifies summation-order differences on near-zero
gradients; the test bound is 1e-6 for SGD and 1e-4 for Adam).

Not provided: multiple machines (helpers are local; cross-host workers in ADR0071 run
whole jobs), CUDA/NCCL, speedups on this small fixture (no throughput claim), fault
tolerance or elastic membership, distributed tabular/RL/agent runs.

Verification: 4 pytest cases (2-worker SGD and 3-worker Adam equivalence with real
spawned processes; Keras/JAX and CUDA refusals) and the training browser journey
running a 2-worker job from the Train tab.
