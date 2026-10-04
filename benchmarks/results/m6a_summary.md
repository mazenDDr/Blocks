# Benchmark summary (generated from the raw file; do not edit)

Raw data: every timed call of every process is in the raw JSON. 5 fresh-process repetitions x 50 timed calls (after 5 warmup calls) per (backend, implementation, workload); batch 16; plain SGD lr 0.01.

Environment: Apple M4 Pro (12 logical CPUs, 8 performance), macOS-27.0-arm64-arm-64bit-Mach-O, Python 3.13.12, torch 2.14.1, tensorflow 2.21.0, keras 3.15.1, jax 0.11.2, jaxlib 0.11.2, numpy 2.5.3; commit 56745669ff (working tree dirty at run time); 2026-10-04T11:04:07Z. Threads requested: 4 (torch, TensorFlow); the JAX flag was requested, its effect unverified. run on a developer laptop with other processes alive (including a resident uvicorn); no isolation, no frequency pinning.

## Reference CNN (VISION 8.1), batch 16, 3x64x64: warm forward (loss only), CPU

Per-process median of the timed calls, then the median (min-max) of those medians across processes, in ms. Ratio = lowered / native using the medians; difference = absolute overhead in ms. Spread = (max - min) / median across processes, per implementation.

| Backend | native median (min-max) | lowered median (min-max) | ratio | difference (ms) | native spread | lowered spread |
|---|---|---|---|---|---|---|
| pytorch | 9.322 (8.819-9.805) | 9.394 (9.198-9.514) | 1.008 | +0.072 | 11% | 3% |
| keras | 3.412 (3.397-3.449) | 3.676 (3.658-3.872) | 1.077 | +0.264 | 2% | 6% |
| jax | 1.512 (1.506-1.525) | 1.530 (1.509-1.551) | 1.012 | +0.019 | 1% | 3% |

## Reference CNN (VISION 8.1), batch 16, 3x64x64: warm train step (forward + backward + SGD), CPU

Per-process median of the timed calls, then the median (min-max) of those medians across processes, in ms. Ratio = lowered / native using the medians; difference = absolute overhead in ms. Spread = (max - min) / median across processes, per implementation.

| Backend | native median (min-max) | lowered median (min-max) | ratio | difference (ms) | native spread | lowered spread |
|---|---|---|---|---|---|---|
| pytorch | 17.411 (17.204-18.109) | 17.302 (17.223-17.737) | 0.994 | -0.109 | 5% | 3% |
| keras | 13.506 (13.386-13.659) | 14.094 (13.864-14.265) | 1.044 | +0.588 | 2% | 3% |
| jax | 7.814 (7.581-7.918) | 7.900 (7.778-8.223) | 1.011 | +0.086 | 4% | 6% |

## MSE fixture (3 elements; absolute overhead of a 4-node graph): warm forward (loss only), CPU

Per-process median of the timed calls, then the median (min-max) of those medians across processes, in ms. Ratio = lowered / native using the medians; difference = absolute overhead in ms. Spread = (max - min) / median across processes, per implementation.

| Backend | native median (min-max) | lowered median (min-max) | ratio | difference (ms) | native spread | lowered spread |
|---|---|---|---|---|---|---|
| pytorch | 0.003 (0.003-0.004) | 0.009 (0.008-0.009) | 2.590 | +0.005 | 8% | 8% |
| keras | 0.035 (0.032-0.036) | 0.066 (0.053-0.066) | 1.849 | +0.030 | 10% | 20% |
| jax | 0.003 (0.003-0.003) | 0.003 (0.003-0.004) | 1.153 | +0.000 | 11% | 12% |

## MSE fixture (3 elements; absolute overhead of a 4-node graph): warm loss + input gradient, CPU

Per-process median of the timed calls, then the median (min-max) of those medians across processes, in ms. Ratio = lowered / native using the medians; difference = absolute overhead in ms. Spread = (max - min) / median across processes, per implementation.

| Backend | native median (min-max) | lowered median (min-max) | ratio | difference (ms) | native spread | lowered spread |
|---|---|---|---|---|---|---|
| pytorch | 0.017 (0.015-0.018) | 0.024 (0.023-0.024) | 1.386 | +0.007 | 14% | 6% |
| keras | 0.036 (0.033-0.044) | 0.067 (0.066-0.070) | 1.880 | +0.031 | 32% | 6% |
| jax | 0.003 (0.003-0.003) | 0.004 (0.004-0.005) | 1.646 | +0.002 | 3% | 12% |

## Cold start (median across processes, seconds)

Import = process start to libraries imported; build = construct the model (lowered: validate + lower + copy weights); first forward / first train step include tracing and compilation.

| Workload | Backend | Impl | import | build | first forward | first step |
|---|---|---|---|---|---|---|
| cnn | pytorch | native | 0.390 | 0.313 | 0.010 | 0.025 |
| cnn | pytorch | lowered | 0.446 | 1.201 | 0.011 | 0.028 |
| cnn | keras | native | 2.059 | 0.014 | 0.028 | 0.052 |
| cnn | keras | lowered | 2.091 | 0.830 | 0.167 | 0.224 |
| cnn | jax | native | 0.229 | 0.033 | 0.056 | 0.147 |
| cnn | jax | lowered | 0.263 | 2.151 | 0.050 | 0.142 |
| mse | pytorch | native | 0.390 | 0.000 | 0.000 | 0.000 |
| mse | pytorch | lowered | 0.446 | 0.001 | 0.000 | 0.000 |
| mse | keras | native | 2.059 | 0.000 | 0.011 | 0.013 |
| mse | keras | lowered | 2.091 | 0.002 | 0.005 | 0.015 |
| mse | jax | native | 0.229 | 0.006 | 0.009 | 0.009 |
| mse | jax | lowered | 0.263 | 0.016 | 0.008 | 0.000 |

## Process peak RSS (MB, median; whole process incl. libraries and both workloads; not a model-memory measurement)

| Backend | native | lowered |
|---|---|---|
| pytorch | 515 | 653 |
| keras | 661 | 881 |
| jax | 564 | 870 |

## Reading these numbers

VISION 18.3 proposes a warm training overhead of at most 1.05x native as a TARGET; the train-step ratios above are the measurement against it, with the spreads shown. These are single-machine laptop runs without isolation: differences smaller than the spread are not distinguishable. Accelerator memory, GPU, large models, the tabular pipeline and the language-model workflow are NOT measured here.
