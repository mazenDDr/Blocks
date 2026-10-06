"""Data-parallel model-graph training across worker processes on one machine (ADR 0073).

The worker process is rank 0; it spawns `workers - 1` helper processes that join a gloo process group on loopback. Every rank lowers
the same graph, loads the same image folder and derives the same data order from the seed; rank 0 broadcasts its initial parameters.
For each global batch every rank computes the SUM of the cross-entropy over its shard, gradients are summed with all_reduce and divided
by the global batch size, and every rank applies the same optimizer step. That equals single-process training on the same batches up
to floating-point summation order, independent of shard sizes. Only rank 0 evaluates and records events and checkpoints.
"""
from __future__ import annotations

import socket
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

STOP, BATCH = 0, 1


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _optimizer(model, cfg):
    return (torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum) if cfg.optimizer == "sgd"
            else torch.optim.Adam(model.parameters(), lr=cfg.lr))


def _shard_grads(model, data, idx: torch.Tensor, rank: int, world: int) -> float:
    """Sum-reduced loss on this rank's shard; gradients become the global batch mean after the all_reduce. Returns the global mean loss."""
    shard = idx.tensor_split(world)[rank]
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    if len(shard):
        loss = F.cross_entropy(model(data.x_train[shard]), data.y_train[shard], reduction="sum")
        loss.backward()
        total = loss.detach().clone()
    else:
        total = torch.zeros(())
    for p in model.parameters():
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= len(idx)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return float(total) / len(idx)


def _broadcast_command(cmd: int, epoch: int = 0, start: int = 0) -> tuple[int, int, int]:
    t = torch.tensor([cmd, epoch, start], dtype=torch.int64)
    dist.broadcast(t, src=0)
    return int(t[0]), int(t[1]), int(t[2])


def _order(cfg, n: int, epoch: int) -> torch.Tensor:
    return torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed * 100003 + epoch))


def _helper(rank: int, world: int, port: int, graph_json: dict[str, Any], cfg_json: dict[str, Any]) -> None:
    from graph_core.schema import Graph
    from graph_core.validate import require_executable
    from graph_core.lower import lower_graph

    from .dataset import load_image_folder
    from .train import RunConfig, _io_contract

    torch.set_num_threads(1)
    cfg = RunConfig.model_validate(cfg_json)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    try:
        graph = Graph.model_validate(graph_json)
        report = require_executable(graph)
        model = lower_graph(graph, report)
        height, _ = _io_contract(graph, report, model)
        width = report.output_types[model.input_ids[0]]["value"].shape[3]
        data = load_image_folder(cfg.data, height, width, cfg.effective_split_seed, cfg.val_fraction)
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        opt = _optimizer(model, cfg)
        model.train()
        n = len(data.x_train)
        while True:
            cmd, epoch, start = _broadcast_command(STOP)
            if cmd == STOP:
                return
            idx = _order(cfg, n, epoch)[start:start + cfg.batch_size]
            _shard_grads(model, data, idx, rank, world)
            opt.step()
    finally:
        dist.destroy_process_group()


class DataParallel:
    """Rank 0 side: start helpers, share the initial parameters, then compute each global batch's gradient together."""

    def __init__(self, graph, cfg, model, data):
        self.cfg, self.model, self.data, self.world = cfg, model, data, cfg.workers
        port = _free_port()
        ctx = mp.get_context("spawn")
        self.procs = [ctx.Process(target=_helper, args=(r, self.world, port, graph.to_json(), cfg.model_dump()), daemon=True)
                      for r in range(1, self.world)]
        for p in self.procs:
            p.start()
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=self.world)
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

    def batch_loss(self, epoch: int, start: int, idx: torch.Tensor) -> float:
        _broadcast_command(BATCH, epoch, start)
        return _shard_grads(self.model, self.data, idx, 0, self.world)

    def close(self) -> None:
        try:
            _broadcast_command(STOP)
        finally:
            dist.destroy_process_group()
            for p in self.procs:
                p.join(timeout=30)
                if p.is_alive():
                    p.kill()
