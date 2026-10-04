"""The common execution interface of the three backends. Everything crossing it is a NumPy array in GRAPH layout (NCHW, PyTorch weight
layout); a backend converts at its own boundary and records every conversion in its compatibility report."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

Params = dict[str, dict[str, np.ndarray]]  # owner node -> param name -> array (graph layout)


class BackendError(Exception):
    """A backend refused or failed before/while executing. `code` is stable."""

    def __init__(self, code: str, message: str, report=None):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.report = code, message, report


@dataclass
class ForwardResult:
    outputs: dict[str, np.ndarray]  # terminal nodes
    activations: dict[str, np.ndarray] = field(default_factory=dict)  # every node, only when capture=True


class Executable:
    """A graph compiled for one backend. Subclasses: TorchExecutable, KerasExecutable, JaxExecutable."""

    backend: str
    input_ids: list[str]
    output_ids: list[str]

    def init_info(self) -> str:  # the declared initialization
        raise NotImplementedError

    def get_params(self) -> Params:
        raise NotImplementedError

    def set_params(self, params: Params) -> None:
        raise NotImplementedError

    def forward(self, inputs: Mapping[str, np.ndarray], capture: bool = False) -> ForwardResult:
        raise NotImplementedError

    def loss_and_grads(self, inputs: Mapping[str, np.ndarray], loss_node: str | None = None, wrt_inputs: tuple[str, ...] = ()):
        """-> (loss: float ndarray, param grads: Params, input grads: {input id: array}). `loss_node` defaults to the single output."""
        raise NotImplementedError

    def sgd_step(self, inputs: Mapping[str, np.ndarray], lr: float, loss_node: str | None = None) -> np.ndarray:
        """One plain SGD update (p <- p - lr * grad, no momentum, no weight decay); returns the loss BEFORE the update."""
        raise NotImplementedError

    def _loss_node(self, loss_node: str | None) -> str:
        if loss_node is not None:
            if loss_node not in self.output_ids and loss_node not in getattr(self, "plan").by_id:
                raise BackendError("E_BACKEND_LOSS_NODE", f"unknown loss node '{loss_node}'")
            return loss_node
        if len(self.output_ids) != 1:
            raise BackendError("E_BACKEND_LOSS_NODE", f"the graph has several outputs {self.output_ids}; name the loss node")
        return self.output_ids[0]
