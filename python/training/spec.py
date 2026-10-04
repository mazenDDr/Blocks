"""The declared training procedure (VISION 10): an ordered stage list plus the configuration of each stage.

The stage list is part of the semantics: the executor runs the stages in the listed order, so editing the order changes behaviour,
and orders that are wrong (clipping after the optimizer step, zeroing gradients between backward and step, ...) are rejected by
`check_procedure` with a stable error code before anything runs."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from graph_core.types import Diagnostic, Fix

STAGE_KINDS = ("zero_grad", "forward", "loss", "backward", "clip", "optimizer_step", "scheduler_step")
DEFAULT_STAGES = ["zero_grad", "forward", "loss", "backward", "clip", "optimizer_step", "scheduler_step"]


class _S(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Cadence(_S):
    unit: Literal["epoch", "optimizer_step"] = "epoch"
    n: int = Field(1, gt=0)


class DataSpec(_S):
    kind: Literal["synthetic_sequence", "synthetic_regression", "image_folder", "provided"] = "synthetic_sequence"
    path: str = ""  # image_folder: root/<class>/<image>; resolved relative to the backend's working directory
    image_size: tuple[int, int] = (64, 64)
    val_fraction: float = Field(0.2, gt=0, lt=1)
    n_train: int = Field(256, gt=0)
    n_val: int = Field(64, gt=0)
    seq_len: int = Field(8, gt=0)
    vocab: int = Field(12, gt=2)
    features: int = Field(4, gt=0)
    outputs: int = Field(2, gt=0)
    data_seed: int = 0
    batch_size: int = Field(16, gt=0)
    shuffle: bool = True
    drop_last: bool = False
    inputs: dict[str, str] = Field(default_factory=dict)  # model input node id -> dataset field (default: the single input gets "x")
    target: str = "y"  # dataset field used as the loss target


class LossSpec(_S):
    kind: Literal["cross_entropy", "mse", "module"] = "cross_entropy"
    module: str = ""
    version: str = "1.0.0"
    args: dict[str, Any] = Field(default_factory=dict)
    output: str = "loss"  # which module output is the scalar loss
    # module input name -> "output" (the model output) or a dataset field name (e.g. "y", "mask", "weights")
    ports: dict[str, str] = Field(default_factory=lambda: {"pred": "output", "target": "y"})


class OptimizerSpec(_S):
    kind: Literal["sgd", "adam", "adamw"] = "sgd"
    lr: float = Field(0.05, gt=0)
    momentum: float = Field(0.0, ge=0)
    nesterov: bool = False
    dampening: float = Field(0.0, ge=0)
    weight_decay: float = Field(0.0, ge=0)
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = Field(1e-8, gt=0)
    amsgrad: bool = False


class ClipSpec(_S):
    kind: Literal["none", "norm", "value"] = "none"
    max_norm: float = Field(1.0, gt=0)
    norm_type: float = 2.0
    value: float = Field(1.0, gt=0)


class SchedulerSpec(_S):
    kind: Literal["none", "step", "exponential", "cosine", "linear_warmup", "reduce_on_plateau"] = "none"
    timing: Literal["optimizer_step", "epoch", "validation"] = "optimizer_step"  # when .step() is called
    step_size: int = Field(10, gt=0)
    gamma: float = Field(0.5, gt=0)
    t_max: int = Field(100, gt=0)
    warmup_steps: int = Field(5, ge=0)
    factor: float = Field(0.5, gt=0, lt=1)
    patience: int = Field(2, ge=0)
    metric: str = "val_loss"
    mode: Literal["min", "max"] = "min"


class AccumulationSpec(_S):
    steps: int = Field(1, ge=1)
    # the loss of each micro-batch is multiplied by 1/window_length ("mean": the sum of micro-batch means equals the mean over the
    # window when the micro-batches have equal size), or left unscaled ("sum")
    normalize: Literal["mean", "sum"] = "mean"
    partial_window: Literal["step", "drop"] = "step"  # a last window shorter than `steps` at the end of an epoch: still step (scaled by its real length) or drop it


class ValidationSpec(_S):
    every: Cadence = Cadence()
    eval_mode: bool = True  # model.eval() during validation (dropout/batch-norm in 'follow' mode behave as in inference)
    no_grad: bool = True  # torch.no_grad(): separate from eval mode


class BestSpec(_S):
    metric: str = "val_loss"
    mode: Literal["min", "max"] = "min"


class CheckpointSpec(_S):
    every: Cadence | None = Cadence(unit="epoch", n=1)
    keep_last: int | None = Field(None, ge=1)
    best: BestSpec | None = None


class EarlyStopSpec(_S):
    metric: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    patience: int = Field(3, ge=1)  # validations without improvement
    min_delta: float = Field(0.0, ge=0)


class WatchSpec(_S):
    param: str  # state-dict style parameter name, e.g. "head.weight"
    index: list[int] = Field(default_factory=lambda: [0])  # element index inside the tensor


class Breakpoint(_S):
    id: str
    kind: Literal["nonfinite_loss", "nonfinite_node", "grad_norm_above", "loss_above", "step_reached"] = "nonfinite_loss"
    node: str | None = None
    threshold: float | None = None
    step: int | None = None


class ProbeSpec(_S):
    node: str
    values: int = 0


class CaptureSpec(_S):
    """Which optimizer steps to record activations/gradients for, and which nodes (None = all)."""

    steps: list[int] = Field(default_factory=list)
    nodes: list[str] | None = None
    max_elements: int = 400_000


class ProcedureSpec(_S):
    seed: int = 0
    deterministic: bool = True
    epochs: int = Field(2, gt=0)
    max_optimizer_steps: int | None = Field(None, ge=1)
    data: DataSpec = DataSpec()
    loss: LossSpec = LossSpec()
    optimizer: OptimizerSpec = OptimizerSpec()
    clip: ClipSpec = ClipSpec()
    scheduler: SchedulerSpec = SchedulerSpec()
    accumulation: AccumulationSpec = AccumulationSpec()
    validation: ValidationSpec = ValidationSpec()
    checkpoint: CheckpointSpec = CheckpointSpec()
    early_stopping: EarlyStopSpec | None = None
    stages: list[str] = Field(default_factory=lambda: list(DEFAULT_STAGES))
    frozen: list[str] = Field(default_factory=list)  # node paths whose parameters are not trained
    watch: WatchSpec | None = None
    breakpoints: list[Breakpoint] = Field(default_factory=list)
    probes: list[ProbeSpec] = Field(default_factory=list)
    assertions: bool = False  # enable diag.assert blocks (execution-changing)
    capture: CaptureSpec | None = None

    @model_validator(mode="after")
    def _kinds(self):
        for s in self.stages:
            if s not in STAGE_KINDS:
                raise ValueError(f"unknown stage '{s}' (known: {list(STAGE_KINDS)})")
        return self


def check_procedure(spec: ProcedureSpec) -> list[Diagnostic]:
    """Order-of-operations rules. Each violated rule is an error with a stable code, the stage involved, and (where there is one) a fix."""
    out: list[Diagnostic] = []

    def err(code, msg, stage=None, fixes=None, sev="error"):
        out.append(Diagnostic(code, msg, sev, stage, None, "/stages", fixes or []))  # type: ignore[arg-type]

    st = spec.stages
    for k in ("forward", "loss", "backward", "optimizer_step"):
        if st.count(k) != 1:
            err("E_PROC_STAGE", f"The procedure needs exactly one '{k}' stage (has {st.count(k)}).", k)
    for k in ("zero_grad", "clip", "scheduler_step"):
        if st.count(k) > 1:
            err("E_PROC_STAGE", f"The procedure has '{k}' {st.count(k)} times; at most once.", k)
    if out:
        return out
    ix = {k: st.index(k) for k in set(st)}
    if not ix["forward"] < ix["loss"] < ix["backward"]:
        err("E_PROC_ORDER", "forward must come before loss, and loss before backward.", "backward")
    if ix["backward"] > ix["optimizer_step"]:
        err("E_PROC_ORDER", "optimizer_step before backward would step on stale or missing gradients.", "optimizer_step")
    if "clip" in ix and not ix["backward"] < ix["clip"] < ix["optimizer_step"]:
        err("E_PROC_ORDER", "clip must come after backward and before optimizer_step: clipping after the step has no effect on the update, and clipping before backward clips nothing.",
            "clip", [Fix("Move clip between backward and optimizer_step")])
    if "zero_grad" in ix:
        z = ix["zero_grad"]
        if ix["forward"] < z < ix["optimizer_step"]:
            err("E_PROC_ORDER", "zero_grad between forward and optimizer_step would erase gradients that the step needs (or, with accumulation, the accumulated sum). "
                "Place it before forward (start of the window) or after optimizer_step (end of the window).", "zero_grad", [Fix("Move zero_grad before forward")])
    else:
        err("W_PROC_NO_ZERO_GRAD", "There is no zero_grad stage: gradients accumulate across optimizer steps forever (that is what will run).", "zero_grad", sev="warning")
    if "scheduler_step" in ix:
        if spec.scheduler.kind == "none":
            err("W_PROC_SCHEDULER_NONE", "A scheduler_step stage exists but no scheduler is configured: it does nothing.", "scheduler_step", sev="warning")
        elif spec.scheduler.timing == "optimizer_step" and ix["scheduler_step"] < ix["optimizer_step"]:
            err("E_PROC_ORDER", "scheduler_step before optimizer_step skips the first learning-rate value (PyTorch warns about this order).", "scheduler_step",
                [Fix("Move scheduler_step after optimizer_step")])
    if spec.scheduler.kind != "none" and spec.scheduler.timing == "optimizer_step" and "scheduler_step" not in ix:
        err("E_PROC_STAGE", "A scheduler with timing 'optimizer_step' is configured but there is no scheduler_step stage, so the learning rate would never change.", "scheduler_step",
            [Fix("Add scheduler_step after optimizer_step")])
    if spec.scheduler.kind != "none" and spec.scheduler.timing != "optimizer_step" and "scheduler_step" in ix:
        err("W_PROC_SCHEDULER_STAGE_IGNORED", f"The scheduler timing is '{spec.scheduler.timing}', so the scheduler_step stage in the list is ignored (the scheduler steps on that event instead).",
            "scheduler_step", sev="warning")
    if spec.scheduler.kind == "reduce_on_plateau" and spec.scheduler.timing != "validation":
        err("E_PROC_STAGE", "reduce_on_plateau reacts to a metric: its timing must be 'validation'.", "scheduler_step")
    if spec.clip.kind != "none" and "clip" not in ix:
        err("E_PROC_STAGE", f"Clipping '{spec.clip.kind}' is configured but there is no clip stage, so nothing would be clipped.", "clip", [Fix("Add clip after backward")])
    if spec.clip.kind == "none" and "clip" in ix:
        err("W_PROC_CLIP_NONE", "A clip stage exists with kind 'none': it does nothing.", "clip", sev="warning")
    if spec.loss.kind == "module" and not spec.loss.module:
        err("E_PROC_LOSS", "The loss is a module but no module is named.", "loss")
    if spec.scheduler.kind == "reduce_on_plateau" and spec.scheduler.metric not in ("val_loss", "val_acc", "train_loss"):
        err("E_PROC_STAGE", f"Unknown scheduler metric '{spec.scheduler.metric}'.", "scheduler_step")
    return out
