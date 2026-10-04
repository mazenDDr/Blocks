from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

HARD_MAX_TRIALS = 200
RESERVED_RUN_FIELDS = {"data", "project_id", "kind", "trial", "source_pins"}


class _S(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Target(_S):
    scope: Literal["node", "run"]
    node: str | None = None
    field: str = Field(..., min_length=1, max_length=120, description="config field (dotted path into nested config) of the node, or run-config field")

    @property
    def key(self) -> str:
        return f"{self.node}.{self.field}" if self.scope == "node" else f"run.{self.field}"


class Range(_S):
    low: float
    high: float
    type: Literal["float", "int"] = "float"
    scale: Literal["linear", "log"] = "linear"


class Variable(_S):
    target: Target
    values: list[Any] | None = Field(None, description="discrete values (grid: all of them; random: sampled uniformly)")
    range: Range | None = Field(None, description="interval (random search only)")
    labels: list[str] | None = Field(None, description="optional names for the discrete values (e.g. ablation variants)")


class Search(_S):
    method: Literal["single", "grid", "random"] = "single"
    variables: list[Variable] = Field(default_factory=list)
    random_trials: int = Field(0, ge=0, le=HARD_MAX_TRIALS)
    sampler_seed: int = 0


class Repeats(_S):
    seeds: list[int] = Field(default_factory=list, max_length=50)
    seed_target: Target | None = Field(None, description="where the seed goes; default: the run-config field 'seed'")
    folds: int | None = Field(None, ge=2, le=50)
    fold_node: str | None = Field(None, description="a tabular.train_validation_split node that receives n_folds/fold")


class MetricSpec(_S):
    name: str = Field(..., description="tabular: a metric of the metrics node (e.g. r2, rmse); model: val_loss | val_acc | train_loss")
    node: str | None = Field(None, description="tabular: the sklearn.metrics node")
    select: Literal["last", "min", "max"] = Field("last", description="model runs: which epoch_end value counts")


class Objective(_S):
    metric: MetricSpec
    direction: Literal["minimize", "maximize"]


class Limits(_S):
    max_trials: int = Field(..., ge=1, le=HARD_MAX_TRIALS)
    max_concurrency: int = Field(1, ge=1, le=1, description="trials run one at a time")
    max_attempts: int = Field(3, ge=1, le=5)


class StudyCreate(_S):
    name: str = Field(..., min_length=1, max_length=120)
    hypothesis: str = Field("", max_length=2000)
    notes: str = Field("", max_length=5000)
    projectId: str | None = None
    graph: dict[str, Any] | None = None
    run_config: dict[str, Any] = Field(default_factory=dict)
    objective: Objective
    search: Search = Field(default_factory=Search)
    repeats: Repeats = Field(default_factory=Repeats)
    limits: Limits
    include_baseline: bool = True
    start: bool = True
