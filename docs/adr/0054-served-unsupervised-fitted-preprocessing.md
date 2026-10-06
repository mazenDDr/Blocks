# ADR0054: serving unsupervised estimators behind fitted preprocessing

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded separately in HANDOFF§51.

ADR0020 serves k-means, Gaussian mixtures and PCA only when requests can carry the
estimator's numeric features directly. A tabular path that imputes, one-hot encodes
or standardizes with a fitted transform before the estimator was refused, so a
clustering placed after the same preprocessing as a supervised model could not be
deployed.

Decision: a separate module, `production/unsup_fitted.py`, captures and serves
those paths. The ADR0020 adapter is not edited: its own source is part of every
registered unsupervised version's implementation hash, and changing it would make
existing versions refuse to serve. That adapter still records its refusal; the new
capture records an `unsup_fitted_pipeline` beside it and registration prefers it.

- Capture walks the recorded table path the way the supervised adapter (ADR0012)
  does: `apply_transform` keeps its native FitState (stored as a trusted worker
  artifact), `select_columns` is replayed, row filters/split/profile are recorded
  as training-only, and the path must end at a source. Anything else refuses with
  a recorded reason. Paths with no fitted step stay with ADR0020.
- Column selections are pruned to the columns something downstream needs, worked
  backwards from the estimator's features through each fitted step, so requests
  never carry unused columns (such as a supervised target selected earlier).
- The input schema is the raw source columns with their types; columns the pinned
  imputer fills are nullable. Requests are validated with the supervised rules.
- Inference replays the pinned steps through the same native operations, then
  uses the run's own scaler and estimator with the ADR0020 output logic (distances,
  responsibilities, scores, reconstruction error). Nothing is refitted.
- Versions keep adapter `unsup` and family = method; the runtime chooses the
  fitted pipeline from the manifest. The reference sample is raw source rows of the
  fitted rows; monitoring compares numeric inputs by distribution and categorical
  inputs by category; reference-input describes the raw contract; the editor's
  version panel lists the replayed steps and estimator features.
- Identity: environment plus hashes of this module, the tabular operations and the
  ADR0020 implementation files.

Not provided: DBSCAN/t-SNE (still cannot map new points), custom code or other
table operations upstream, target-dependent transforms, and Keras/JAX serving.

Verification: a real tabular worker run of the regression example's
impute → one-hot → standardize path feeding k-means with a one-hot feature, then
HTTP registration, release and serving: the manifest has the expected steps,
nullable raw schema and no target column; 40 raw reference rows (including a null
that the pinned imputer fills) give exactly the run's own native cluster
assignments and distances on the recorded transformed rows; extra columns and
wrong types refuse; categorical drift monitoring works; ADR0020's implementation
file list is unchanged and all earlier unsupervised tests pass. A scratch owned
Chrome check showed the version panel with the replayed steps.
