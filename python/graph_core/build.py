"""A small programmatic builder for graphs and module definitions (used by tests and the example generators).
It only creates the same JSON-able Graph / ModuleDef objects the editor creates; nothing here is executed."""
from __future__ import annotations

from typing import Any

from .schema import Edge, Endpoint, Graph, ModuleDef, ModuleOutput, ModuleParam, Node, PortSpec


def ep(ref: str, default_port: str) -> Endpoint:
    node, _, port = ref.partition(".")
    return Endpoint(node=node, port=port or default_port)


class Body:
    """Nodes and edges shared by Graph and ModuleDef building."""

    def __init__(self):
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []

    def node(self, id: str, type: str, shared_with: str | None = None, **config: Any) -> str:
        self.nodes.append(Node(id=id, type=type, config=config, sharedWith=shared_with))
        return id

    def wire(self, src: str, dst: str, kind: str = "tensor") -> None:
        """wire("conv.output", "relu.input"); ports default to output / input; "$in.x" is a module input."""
        s, d = ep(src, "output"), ep(dst, "input")
        if "." not in src and any(n.id == s.node and n.type in ("core.tensor_input", "tensor.constant") for n in self.nodes):
            s = Endpoint(node=s.node, port="value")
        self.edges.append(Edge(id=f"{s.node}_{s.port}__{d.node}_{d.port}", kind=kind, **{"from": s}, to=d))

    def chain(self, *ids: str) -> None:
        for a, b in zip(ids, ids[1:]):
            self.wire(a, b)

    def instance(self, id: str, module: ModuleDef | str, version: str | None = None, share: str = "clone", **args: Any) -> str:
        mid, ver = (module.id, module.version) if isinstance(module, ModuleDef) else (module, version or "1.0.0")
        return self.node(id, "core.composite", module=mid, version=ver, args=args, share=share)


class GraphBuilder(Body):
    def __init__(self):
        super().__init__()
        self.modules: list[ModuleDef] = []

    def input(self, id: str, shape: list, dtype: str = "float32") -> str:
        return self.node(id, "core.tensor_input", shape=shape, dtype=dtype)

    def build(self) -> Graph:
        return Graph(nodes=self.nodes, edges=self.edges, modules=self.modules)


class ModuleBuilder(Body):
    def __init__(self, id: str, version: str = "1.0.0", description: str = ""):
        super().__init__()
        self.id, self.version, self.description = id, version, description
        self.inputs: list[PortSpec] = []
        self.outputs: list[ModuleOutput] = []
        self.params: list[ModuleParam] = []
        self.reduction: dict | None = None

    def in_(self, name: str, shape: list | None = None, dtype: str = "float32") -> str:
        self.inputs.append(PortSpec(name=name, dtype=dtype, shape=shape))
        return f"$in.{name}"

    def out(self, name: str, ref: str, shape: list | None = None, dtype: str | None = None) -> None:
        self.outputs.append(ModuleOutput(name=name, **{"from": ep(ref, "output")}, shape=shape, dtype=dtype))

    def param(self, name: str, default: Any, type: str | None = None) -> dict:
        self.params.append(ModuleParam(name=name, default=default, type=type))
        return {"$param": name}

    def build(self) -> ModuleDef:
        return ModuleDef(id=self.id, version=self.version, description=self.description, inputs=self.inputs, outputs=self.outputs,
                         params=self.params, nodes=self.nodes, edges=self.edges, reduction=self.reduction)


def residual_block(channels: int = 8, version: str = "1.0.0") -> ModuleDef:
    """conv_a -> relu_a -> conv_b -> add(skip) -> relu_out; the skip is the block input (same channels, 3x3 'same' padding)."""
    m = ModuleBuilder("residual_block", version, "y = relu(x + conv_b(relu(conv_a(x)))): a residual connection around two 3x3 convolutions")
    x = m.in_("x", [None, channels, None, None])
    c = m.param("channels", channels, "int")
    m.node("conv_a", "pytorch.nn.conv2d", out_channels=c, kernel_size=[3, 3], padding=[1, 1])
    m.node("relu_a", "pytorch.nn.relu")
    m.node("conv_b", "pytorch.nn.conv2d", out_channels=c, kernel_size=[3, 3], padding=[1, 1])
    m.node("skip_add", "tensor.add")
    m.node("relu_out", "pytorch.nn.relu")
    m.wire(x, "conv_a.input")
    m.chain("conv_a", "relu_a", "conv_b")
    m.wire("conv_b.output", "skip_add.a")
    m.wire(x, "skip_add.b")
    m.wire("skip_add.output", "relu_out.input")
    m.out("y", "relu_out.output", [None, channels, None, None])
    return m.build()


DIVISORS = {
    "element_count": "divide by the number of elements (valid or not): mean of weight*mask*(pred-target)^2",
    "valid_count": "divide by the number of VALID elements (sum of mask)",
    "weight_sum": "divide by the sum of the weights of the valid elements (sum of weight*mask)",
}


def masked_weighted_loss(divisor: str = "valid_count", eps: float = 1e-8, version: str = "1.0.0") -> ModuleDef:
    """loss = sum(weight * mask * (pred - target)^2) / D, built only from primitives. D depends on the declared divisor.
    Empty-denominator policy: D is clamped to at least `eps`; if nothing is valid the numerator is 0, so the loss and its gradient are 0."""
    m = ModuleBuilder(f"masked_weighted_mse_{divisor}", version,
                      f"Masked, weighted squared error. {DIVISORS[divisor]}. If the denominator would be 0 it is clamped to {eps}, giving loss 0 and gradient 0.")
    pred, target = m.in_("pred", ["N", None]), m.in_("target", ["N", None])
    mask, weights = m.in_("mask", ["N", None]), m.in_("weights", ["N", None])
    m.node("diff", "tensor.sub")
    m.node("sq", "core.square")
    m.node("wm", "tensor.mul")  # weight * mask
    m.node("terms", "tensor.mul")  # wm * squared error
    m.wire(pred, "diff.a")
    m.wire(target, "diff.b")
    m.wire("diff.output", "sq.input")
    m.wire(weights, "wm.a")
    m.wire(mask, "wm.b")
    m.wire("sq.output", "terms.a")
    m.wire("wm.output", "terms.b")
    if divisor == "element_count":
        m.node("loss", "core.mean")
        m.wire("terms.output", "loss.input")
    else:
        m.node("num", "core.sum")
        m.wire("terms.output", "num.input")
        m.node("den_raw", "core.sum")
        m.wire(mask if divisor == "valid_count" else "wm.output", "den_raw.input")
        m.node("den", "tensor.clamp", min=eps)
        m.wire("den_raw.output", "den.input")
        m.node("loss", "tensor.div")
        m.wire("num.output", "loss.a")
        m.wire("den.output", "loss.b")
    m.out("loss", "loss.output", [], "float32")
    m.reduction = {"divisor": divisor, "description": DIVISORS[divisor], "empty_policy": f"denominator clamped to >= {eps}; loss 0 when nothing is valid",
                   "formula": "sum(w*m*(p-t)^2) / " + {"element_count": "N_elements", "valid_count": "sum(m)", "weight_sum": "sum(w*m)"}[divisor]}
    return m.build()


# ====================================================================================================== transformer family (VISION 9.3 / 9.4)
def multi_head_attention(d_model: int = 16, heads: int = 2, version: str = "1.0.0") -> ModuleDef:
    """Multi-head self-attention built only from primitives. Inner node ids are part of the module's documented contract, so an inspector can
    find the projections, the scaled scores, the applied mask, the weights and the weighted values by role:
    q_proj/k_proj/v_proj (dense), q_heads/k_heads/v_heads [N,H,T,dh], scores (scaled QK^T), mask_b + masked, weights (softmax), context, merged, out_proj."""
    assert d_model % heads == 0
    dh = d_model // heads
    m = ModuleBuilder("multi_head_attention", version,
                      "Self-attention: Q,K,V = xWq,xWk,xWv split into heads; scores = QK^T/sqrt(d_head); padding mask (False = ignore key) fills -1e9 before softmax; "
                      "weights = softmax over keys; context = weights V; heads merged and projected by Wo.")
    x = m.in_("x", [None, None, d_model])
    mask = m.in_("mask", [None, None], "bool")
    d = m.param("d_model", d_model, "int")
    scale = 1.0 / (dh ** 0.5)
    for n in ("q", "k", "v"):
        m.node(f"{n}_proj", "tensor.dense", out_features=d)
        m.node(f"{n}_split", "tensor.reshape", shape=["N", -1, heads, dh], axes=["N", "T", "H", "dh"])
        m.node(f"{n}_heads", "tensor.permute", dims=[0, 2, 1, 3], axes=["N", "H", "T", "dh"])
        m.wire(x, f"{n}_proj.input")
        m.chain(f"{n}_proj", f"{n}_split", f"{n}_heads")
    m.node("k_t", "tensor.permute", dims=[0, 1, 3, 2], axes=["N", "H", "dh", "T"])
    m.wire("k_heads.output", "k_t.input")
    m.node("raw_scores", "tensor.matmul")
    m.wire("q_heads.output", "raw_scores.a")
    m.wire("k_t.output", "raw_scores.b")
    m.node("scores", "core.scalar_mul", factor=scale)
    m.wire("raw_scores.output", "scores.input")
    m.node("mask_b", "tensor.reshape", shape=["N", 1, 1, -1], axes=["N", "1", "1", "T_k"])
    m.wire(mask, "mask_b.input")
    m.node("neg_inf", "tensor.constant", value=-1e9, label="masked positions get a large negative score before softmax")
    m.node("masked", "tensor.where")
    m.wire("mask_b.output", "masked.cond")
    m.wire("scores.output", "masked.a")
    m.wire("neg_inf.value", "masked.b")
    m.node("weights", "tensor.softmax", dim=-1)
    m.wire("masked.output", "weights.input")
    m.node("attn_drop", "tensor.dropout", p=m.param("attn_dropout", 0.0, "float"))
    m.wire("weights.output", "attn_drop.input")
    m.node("context", "tensor.matmul")
    m.wire("attn_drop.output", "context.a")
    m.wire("v_heads.output", "context.b")
    m.node("ctx_perm", "tensor.permute", dims=[0, 2, 1, 3], axes=["N", "T", "H", "dh"])
    m.node("merged", "tensor.reshape", shape=["N", -1, d_model], axes=["N", "T", "D"])
    m.node("out_proj", "tensor.dense", out_features=d)
    m.chain("context", "ctx_perm", "merged", "out_proj")
    m.out("y", "out_proj.output", [None, None, d_model])
    m.reduction = None
    return m.build()


def transformer_encoder_block(d_model: int = 16, heads: int = 2, d_ff: int = 32, version: str = "1.0.0") -> ModuleDef:
    """Post-LN encoder block (Add & Norm after each sub-layer, as in the original Transformer): x1 = LN(x + MHA(x)); y = LN(x1 + FFN(x1))."""
    m = ModuleBuilder("transformer_encoder_block", version,
                      "Encoder block, post-layer-norm: x1 = LayerNorm(x + Attention(x)); y = LayerNorm(x1 + W2 GELU(W1 x1)). Residual paths are the add nodes.")
    x = m.in_("x", [None, None, d_model])
    mask = m.in_("mask", [None, None], "bool")
    m.instance("attn", "multi_head_attention", d_model=d_model)
    m.wire(x, "attn.x")
    m.wire(mask, "attn.mask")
    m.node("add1", "tensor.add")
    m.wire(x, "add1.a")
    m.wire("attn.y", "add1.b")
    m.node("ln1", "tensor.layernorm", normalized_dims=1)
    m.wire("add1.output", "ln1.input")
    m.node("ff1", "tensor.dense", out_features=d_ff)
    m.node("act", "tensor.gelu")
    m.node("ff2", "tensor.dense", out_features=d_model)
    m.chain("ln1", "ff1", "act", "ff2")
    m.node("add2", "tensor.add")
    m.wire("ln1.output", "add2.a")
    m.wire("ff2.output", "add2.b")
    m.node("ln2", "tensor.layernorm", normalized_dims=1)
    m.wire("add2.output", "ln2.input")
    m.out("y", "ln2.output", [None, None, d_model])
    return m.build()


def transformer_classifier(vocab: int = 12, seq_len: int = 8, d_model: int = 16, heads: int = 2, d_ff: int = 32, layers: int = 1, share_layers: bool = False) -> Graph:
    """tokens [N,T] int64 (0 = padding) -> token + learned positional embedding -> encoder block(s) -> masked mean pooling -> 2-class logits."""
    g = GraphBuilder()
    g.modules += [multi_head_attention(d_model, heads), transformer_encoder_block(d_model, heads, d_ff)]
    g.input("tokens", ["N", seq_len], "int64")
    g.node("tok_emb", "tensor.embedding", num_embeddings=vocab, embedding_dim=d_model, padding_idx=0)
    g.node("positions", "tensor.arange", n=seq_len)
    g.node("pos_emb", "tensor.embedding", num_embeddings=seq_len, embedding_dim=d_model)
    g.node("embed_sum", "tensor.add")
    g.wire("tokens", "tok_emb.input")
    g.wire("positions.output", "pos_emb.input")
    g.wire("tok_emb.output", "embed_sum.a")
    g.wire("pos_emb.output", "embed_sum.b")
    g.node("pad_id", "tensor.constant", value=0, dtype="int64")
    g.node("key_mask", "tensor.compare", op="ne")
    g.wire("tokens", "key_mask.a")
    g.wire("pad_id.value", "key_mask.b")
    prev = "embed_sum.output"
    for i in range(layers):
        name = f"enc{i + 1}"
        g.instance(name, "transformer_encoder_block", share=("enc1" if share_layers and i > 0 else "clone"))
        g.wire(prev, f"{name}.x")
        g.wire("key_mask.output", f"{name}.mask")
        prev = f"{name}.y"
    g.node("mask_f", "tensor.cast", dtype="float32")
    g.node("mask_col", "tensor.reshape", shape=["N", -1, 1])
    g.wire("key_mask.output", "mask_f.input")
    g.wire("mask_f.output", "mask_col.input")
    g.node("masked_h", "tensor.mul")
    g.wire(prev, "masked_h.a")
    g.wire("mask_col.output", "masked_h.b")
    g.node("pool_sum", "core.sum", dims=[1])
    g.wire("masked_h.output", "pool_sum.input")
    g.node("count_sum", "core.sum", dims=[1])
    g.wire("mask_col.output", "count_sum.input")
    g.node("count", "tensor.clamp", min=1.0)
    g.wire("count_sum.output", "count.input")
    g.node("pooled", "tensor.div")
    g.wire("pool_sum.output", "pooled.a")
    g.wire("count.output", "pooled.b")
    g.node("head", "tensor.dense", out_features=2)
    g.wire("pooled.output", "head.input")
    return g.build()
