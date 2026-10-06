# ADR0061: serving PyTorch-trained image classifiers on Keras and JAX

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded separately in HANDOFF§58.

Keras 3 and JAX run the portable graph subset (forward, loss, gradients, one SGD
step; ADR0028/0029) but have no worker training runs or checkpoints, and no serving
family used them. Teams that deploy on TensorFlow/Keras or JAX could not serve a
model trained here.

Decision: two serving adapters, `model_keras` and `model_jax`, registered from a
completed PyTorch image-classifier run with node `keras:<output>` or `jax:<output>`,
in their own module `production/portable_model_adapter.py`.

- The version starts from the PyTorch image adapter's manifest (ADR0018): the
  latest complete checkpoint, the training preprocessing, the class names and the
  frozen reference images, with the same source-folder identity check.
- The checkpoint is loaded into the PyTorch lowering and its parameters are copied
  in graph layout into the Keras/JAX executable compiled from the same graph, using
  the existing declared layout conversions. Training never happens on Keras/JAX.
- Registration runs every frozen reference image through both backends and refuses
  (E_PORTABLE_TOLERANCE) unless every logit satisfies the declared float32 forward
  tolerance |backend − pytorch| ≤ atol + rtol·|pytorch| (2e-5, 1e-4) and every
  predicted class matches. The measurement is stored in the manifest.
- Requests, validation, preprocessing, outputs (plus `backend`), labels and
  monitoring are those of the image adapter; only the forward pass runs on the
  other backend, and the PyTorch module is dropped after parameter export.
- Identity: the image adapter's environment/implementation plus this module, the
  backend runtime/spec files and the installed keras/tensorflow/jax/jaxlib
  versions (not the whole requirements file).

Measured on this Mac for the reference CNN (12 frozen images): Keras max logit
difference 1.19e-7, JAX 7.45e-8, all classes equal. Not provided: Keras/JAX
training runs or checkpoints, other model families or graphs outside the portable
subset, float64 JAX, accelerators, or exported SavedModel/StableHLO artifacts.

Verification: for both backends, registration from a real trained reference-CNN
run, the stored agreement measurement, deployment, predictions equal to the PyTorch
version on the same images with probabilities within 1e-4, labels and monitoring; a
zero tolerance makes registration refuse (the gate is real); a changed backend
version refuses serving; unknown backend prefixes are not registered.
