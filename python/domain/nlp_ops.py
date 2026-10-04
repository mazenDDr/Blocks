"""NLP operations of the `domain` graph kind: labelled-corpus source with character spans, subword tokenizer with a declared label alignment policy, token tagger with seqeval-convention evaluation."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any, Literal

import numpy as np
import torch
from pydantic import Field

from graph_core.registry import register
from graph_core.types import Fix, OpError
from nlp import iob, model as M, subword as W
from operations._common import StrictConfig
from tabular.core import ExecutionError, VType, resolve_path

from . import checkpoints as CP
from .core import NLP_REPORT, TEXT_CORPUS, TOKEN_BATCH, DomainOperation, fixture_path, r, sha256_file, value

DEFAULT_JSONL = "examples/fixtures/synthetic_ner.jsonl"
SYNTH = "SYNTHETIC fixture (examples/make_domain_fixtures.py): template sentences with made-up names, not real text."


@lru_cache(maxsize=8)
def _types_of(path: str, mtime: int) -> tuple[list[str], int]:
    types, n = set(), 0
    for line in open(path):
        rec = json.loads(line)
        n += 1
        types.update(s["label"] for s in rec["spans"])
    return sorted(types), n


def read_corpus(path, n: int | None) -> list[dict]:
    recs = []
    for k, line in enumerate(open(path)):
        if not line.strip():
            continue
        rec = json.loads(line)
        L = len(rec["text"])
        for sp in rec["spans"]:
            if not (0 <= sp["start"] < sp["end"] <= L):
                raise ExecutionError("E_NLP_SPAN_RANGE", f"{rec['id']}: span [{sp['start']}, {sp['end']}) is outside the text (length {L})")
        recs.append(rec)
        if n is not None and len(recs) >= n:
            break
    return recs


# ================================================================================================ source
class NlpSourceConfig(StrictConfig):
    path: str = Field(DEFAULT_JSONL, description="JSON-lines corpus: {id, text, spans:[{start,end,label}]} with CHARACTER offsets (end exclusive)")
    n: int | None = Field(None, ge=1, title="use the first n sentences (empty = all)")


@register
class NlpSource(DomainOperation):
    type = "domain.nlp_source"
    inputs = ()
    outputs = ("corpus",)
    out_kinds = {"corpus": TEXT_CORPUS}
    Config = NlpSourceConfig
    summary_kind = "nlp_corpus"

    def infer(self, cfg, ins, node_id):
        p = resolve_path(cfg.path) if cfg.path else None
        if p is None or not p.is_file():
            raise OpError("E_FIXTURE_MISSING", f"Corpus '{cfg.path}' was not found. The committed SYNTHETIC fixture is {DEFAULT_JSONL}.", None, [Fix(f"Set path to {DEFAULT_JSONL}")])
        try:
            types, n = _types_of(str(p), p.stat().st_mtime_ns)
        except Exception as e:  # noqa: BLE001
            raise OpError("E_SOURCE_UNREADABLE", f"Cannot parse '{cfg.path}' as the span corpus format: {e}")
        return {"corpus": VType(TEXT_CORPUS, {"types": types, "n": min(n, cfg.n) if cfg.n else n})}

    def execute(self, cfg, ins, ctx):
        p = fixture_path(cfg.path, "the committed fixture is examples/fixtures/synthetic_ner.jsonl")
        recs = read_corpus(p, cfg.n)
        types = sorted({s["label"] for r_ in recs for s in r_["spans"]})
        counts = {t: sum(1 for r_ in recs for s in r_["spans"] if s["label"] == t) for t in types}
        sha = sha256_file(p)
        data = {"contract": {"n": len(recs), "types": types, "entityCounts": counts, "offsets": "character offsets into the raw text, end exclusive"}, "source": {"path": str(p), "sha256": sha, "synthetic": True}}
        ex = [{"id": x["id"], "text": x["text"], "spans": [{**s, "surface": x["text"][s["start"]:s["end"]]} for s in x["spans"]]} for x in recs[:8]]
        return {"corpus": value(TEXT_CORPUS, data, recs)}, {"path": str(p), "sha256": sha, "synthetic": True, "note": SYNTH, "contract": data["contract"],
                                                            "sentencesWithoutEntities": sum(1 for x in recs if not x["spans"]), "examples": ex,
                                                            "provenance": {"source": "read from the fixture file; every span was checked to lie inside its text"}}

    def explain(self, cfg, inputs, outputs):
        return {"equation": "record = (id, text, spans[(start, end, label)])", "rule": "Spans are character offsets into the raw text. Words, subwords and tags are all derived from them later, never the other way round.", "note": SYNTH}


# ================================================================================================ tokenizer
class TokenizerConfig(StrictConfig):
    fitted_model_id: str | None = Field(None, pattern=r"^[0-9a-f]{64}$", description="Reuse the exact fitted tokenizer from this internal NLP model, without refitting.")
    vocab_size: int = Field(300, ge=40, le=5000, title="WordPiece vocabulary size (trained on the TRAINING sentences only)")
    lowercase: bool = Field(True, title="normalization: lowercase")
    max_length: int = Field(48, ge=8, le=512, title="maximum subwords per sentence including [CLS]/[SEP] (longer sentences are truncated and counted)")
    label_policy: Literal["first_subword", "all_subwords"] = Field("first_subword", title="label alignment policy")
    ignore_index: int = Field(-100, title="label for positions that are not scored (special tokens, padding, and continuation subwords under first_subword)")
    scheme: Literal["iob2"] = "iob2"
    val_fraction: float = Field(0.25, gt=0, lt=1)
    seed: int = 0


@register
class NlpTokenizer(DomainOperation):
    type = "domain.nlp_tokenizer"
    inputs = ("corpus",)
    outputs = ("tokens",)
    in_kinds = {"corpus": TEXT_CORPUS}
    out_kinds = {"tokens": TOKEN_BATCH}
    Config = TokenizerConfig
    summary_kind = "nlp_tokens"

    def infer(self, cfg, ins, node_id):
        if cfg.ignore_index >= 0:
            raise OpError("E_NLP_IGNORE_INDEX_RANGE", f"ignore_index {cfg.ignore_index} would collide with a label id; it must be negative (the default -100 is what torch's cross_entropy ignores).", None, [Fix("Use -100", None, "ignore_index", -100)])
        types = ins["corpus"].info.get("types", [])
        return {"tokens": VType(TOKEN_BATCH, {"labelPolicy": cfg.label_policy, "ignoreIndex": cfg.ignore_index, "scheme": cfg.scheme, "labels": W.tag_list(types), "maxLength": cfg.max_length,
                                              "lowercase": cfg.lowercase})}

    def execute(self, cfg, ins, ctx):
        c = ins["corpus"]
        recs = c.obj
        types = c.data["contract"]["types"]
        tags = W.tag_list(types)
        t2i = {t: i for i, t in enumerate(tags)}
        tr, va = M.split_indices(len(recs), cfg.val_fraction, cfg.seed)
        if cfg.fitted_model_id:
            from tokenizers import Tokenizer
            if ctx.store is None:
                raise ExecutionError("E_DOMAIN_CHECKPOINT_STORE", "Fitted tokenizer reuse requires the owning workbench.")
            m = CP.read_manifest(ctx.store, cfg.fitted_model_id)
            if m["family"] != "nlp" or m["source"] != c.data.get("source"):
                raise ExecutionError("E_DOMAIN_RESUME_INCOMPATIBLE", "Tokenizer model family or corpus identity differs.")
            tok = Tokenizer.from_str(m["inference"]["tokenizerJson"])
            if m["inference"]["tokenizerConfig"] != cfg.model_dump(exclude={"fitted_model_id"}):
                raise ExecutionError("E_DOMAIN_RESUME_INCOMPATIBLE", "Tokenizer settings or corpus split differs.")
        else:
            tok = W.train_tokenizer([recs[i]["text"] for i in tr], cfg.vocab_size, cfg.lowercase)
        try:
            exs = [W.encode_example(tok, x, t2i, cfg.max_length, cfg.label_policy, cfg.ignore_index) for x in recs]
        except W.NlpError as e:
            raise ExecutionError(e.code, e.message) from e
        viol = [{"id": e.id, **v} for e in exs for v in e.violations]
        if viol:
            raise ExecutionError("E_NLP_IOB2_INVALID", f"gold labels derived from the spans violate IOB2: {viol[0]}")
        vocab = tok.get_vocab()
        vsha = hashlib.sha256(json.dumps(sorted(vocab.items()), sort_keys=True).encode()).hexdigest()
        n_words = sum(len(e.words) for e in exs)
        n_sub = sum(len([w for w in e.word_ids if w is not None]) for e in exs)
        data = {"contract": {"n": len(exs), "nTrain": len(tr), "nVal": len(va), "labels": tags, "labelPolicy": cfg.label_policy, "ignoreIndex": cfg.ignore_index, "scheme": cfg.scheme,
                             "maxLength": cfg.max_length, "padId": tok.token_to_id("[PAD]"), "vocabSize": tok.get_vocab_size(), "tokenizerFittedOn": f"{len(tr)} training sentences",
                             "truncatedWords": sum(e.truncated_words for e in exs), "wordsPerSentence": n_words / len(exs), "subwordsPerWord": n_sub / max(n_words, 1)},
                "source": c.data.get("source"), "vocabSha256": vsha, "tokenizerJson": tok.to_str()}
        def rec(e: W.Example, split: str):
            return {"id": e.id, "split": split, "text": e.text, "spans": e.spans, "tokens": e.tokens, "ids": e.ids, "offsets": e.offsets, "wordIds": e.word_ids, "attention": [1] * len(e.ids),
                    "labels": [tags[y] if y != cfg.ignore_index else None for y in e.labels], "labelIds": e.labels, "words": e.words}
        show = [(exs[i], "validation") for i in va[:4]] + [(exs[i], "train") for i in tr[:3]]
        batch_exs = [e for e, _ in show[:4]]
        batch_ids, batch_mask, batch_labels = W.batch_tensors(batch_exs, tok.token_to_id("[PAD]"), cfg.ignore_index)
        padded = []
        for i, e in enumerate(batch_exs):
            extra = batch_ids.shape[1] - len(e.ids)
            padded.append({**rec(e, "validation"), "tokens": e.tokens + ["[PAD]"] * extra,
                           "ids": batch_ids[i].tolist(), "attention": batch_mask[i].tolist(), "labelIds": batch_labels[i].tolist(),
                           "labels": [tags[y] if y != cfg.ignore_index else None for y in batch_labels[i].tolist()],
                           "offsets": e.offsets + [(0, 0)] * extra, "wordIds": e.word_ids + [None] * extra})
        summary = {"batch": {"shape": list(batch_ids.shape), "lengths": batch_mask.sum(1).tolist(), "examples": padded,
                             "note": "Recorded padded batch of up to four validation sentences; attention=0 and ignore_index on padding."},
                   "contract": data["contract"], "policyText": W.POLICY_TEXT[cfg.label_policy], "ignoreIndexMeaning": "positions with this label are skipped by the loss and by evaluation: [CLS]/[SEP]/padding always, continuation subwords under first_subword",
                   "tokenizer": {"model": "WordPiece (Hugging Face tokenizers " + __import__("tokenizers").__version__ + ")", "normalizer": "lowercase" if cfg.lowercase else "none", "preTokenizer": "BertPreTokenizer (whitespace + punctuation)",
                                 "specialTokens": W.SPECIAL, "vocabSize": tok.get_vocab_size(), "vocabSha256": vsha, "fittedOn": {"split": "train", "sentences": len(tr), "seed": cfg.seed},
                                 "sampleVocab": [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])[4:44]]},
                   "examples": [rec(e, s) for e, s in show], "split": {"seed": cfg.seed, "valFraction": cfg.val_fraction, "valIndices": va[:50]},
                   "provenance": {"source": c.data.get("source"), "note": "token offsets are character offsets into the ORIGINAL text; word labels come from the character spans"}}
        return {"tokens": value(TOKEN_BATCH, data, {"examples": exs, "tok": tok, "tags": tags, "tr": tr, "va": va, "cfg": cfg})}, summary

    def explain(self, cfg, inputs, outputs):
        return {"equation": "text -> normalize -> pre-tokenize into words -> WordPiece subwords with character offsets;  word tag from the spans;  subword label = policy(word tag)",
                "rule": W.POLICY_TEXT[cfg.label_policy] + f". Label {cfg.ignore_index} marks positions that are not scored.",
                "note": "The vocabulary is trained on the training sentences only, so validation text cannot influence it. Words whose spans do not start and end on word boundaries are rejected (E_NLP_SPAN_MISALIGNED)."}


# ================================================================================================ tagger
class TaggerConfig(StrictConfig):
    resume_model_id: str | None = Field(None, pattern=r"^[0-9a-f]{64}$", description="Internal model manifest identity to resume at a completed epoch; epochs is the total target.")
    label_policy: Literal["first_subword", "all_subwords"] = Field("first_subword", title="label policy this tagger's loss and read-out assume (must equal the tokenizer's)")
    loss_ignore_index: int = Field(-100, title="cross-entropy ignore_index (must equal the tokenizer's ignore index)")
    eval_mode: Literal["default", "strict"] = Field("default", title="span evaluation convention (seqeval: default = conlleval, strict = IOB2 only)")
    epochs: int = Field(25, ge=1, le=300)
    lr: float = Field(5e-3, gt=0)
    embedding: int = Field(32, ge=4, le=256)
    hidden: int = Field(32, ge=4, le=256)
    batch_size: int = Field(16, ge=1, le=256)
    seed: int = 0
    n_inspect: int = Field(8, ge=1, le=24)


@register
class NlpTagger(DomainOperation):
    type = "domain.nlp_tagger"
    inputs = ("tokens",)
    outputs = ("report",)
    in_kinds = {"tokens": TOKEN_BATCH}
    out_kinds = {"report": NLP_REPORT}
    Config = TaggerConfig
    summary_kind = "nlp_tagger"

    def infer(self, cfg, ins, node_id):
        info = ins["tokens"].info
        if cfg.label_policy != info["labelPolicy"]:
            raise OpError("E_NLP_LABEL_POLICY", f"This tagger assumes label policy '{cfg.label_policy}' but the tokenizer produced '{info['labelPolicy']}'. Labels and the word-level read-out would not line up.", "tokens",
                          [Fix(f"Set label_policy to '{info['labelPolicy']}'", None, "label_policy", info["labelPolicy"])])
        if cfg.loss_ignore_index != info["ignoreIndex"]:
            raise OpError("E_NLP_IGNORE_INDEX", f"The loss ignores label {cfg.loss_ignore_index} but the tokenizer marks unscored positions with {info['ignoreIndex']}: those positions would be trained on as if they were labels.", "tokens",
                          [Fix(f"Set loss_ignore_index to {info['ignoreIndex']}", None, "loss_ignore_index", info["ignoreIndex"])])
        return {"report": VType(NLP_REPORT, {"labels": info["labels"]})}

    def execute(self, cfg, ins, ctx):
        d = ins["tokens"]
        o = d.obj
        exs, tok, tags, tr, va = o["examples"], o["tok"], o["tags"], o["tr"], o["va"]
        signature = CP.fingerprint(cfg.model_dump(exclude={"epochs", "n_inspect", "resume_model_id"}),
            {"data": d.data, "tr": tr, "va": va, "examples": [{"ids": e.ids, "labels": e.labels, "words": e.words} for e in exs]})
        saved = CP.resume(ctx, cfg.resume_model_id, "nlp", signature)
        res = M.train_and_eval(exs, tok, tags, epochs=cfg.epochs, lr=cfg.lr, hidden=cfg.hidden, emb=cfg.embedding, batch_size=cfg.batch_size, seed=cfg.seed, ignore_index=cfg.loss_ignore_index,
                               tr=tr, va=va, n_inspect=cfg.n_inspect, scheme_mode=cfg.eval_mode, resume_state=saved)
        # padding contract: a sentence has the same logits alone and inside a padded batch
        model = res["model"]
        pad_id = tok.token_to_id("[PAD]")
        vax = [exs[i] for i in va]
        ids, mask, _ = W.batch_tensors(vax, pad_id, cfg.loss_ignore_index)
        longest = max(range(len(vax)), key=lambda i: len(vax[i].ids))
        short = min(range(len(vax)), key=lambda i: len(vax[i].ids))
        with torch.no_grad():
            full = model(ids, mask)
            alone = model(ids[short:short + 1, :len(vax[short].ids)], mask[short:short + 1, :len(vax[short].ids)])
        mask_dev = float((full[short, :len(vax[short].ids)] - alone[0]).abs().max())
        try:
            import seqeval
            from seqeval.metrics import f1_score
            cross = {"library": f"seqeval {getattr(seqeval, '__version__', '1.2.2')}", "microF1": f1_score(res["goldWords"], res["predWords"]),
                     "workbenchMicroF1": iob.prf(res["goldWords"], res["predWords"], "default")["micro"]["f1"]}
        except Exception as e:  # noqa: BLE001
            cross = {"error": str(e)}
        sm = res["spanMetrics"]
        summary = {"config": cfg.model_dump(), "nParams": res["nParams"], "curve": res["curve"], "contract": d.data["contract"], "labels": tags,
                   "spanMetrics": {**sm, "provenance": f"entity-level on WORD tags of the {len(va)} validation sentences (prediction read at each word's first subword); seqeval conventions, mode '{cfg.eval_mode}'; tokenizer {d.data['contract']['vocabSize']}-piece WordPiece, labels {d.data['contract']['labelPolicy']}"},
                   "strictMetrics": res["strictMetrics"], "crossCheck": cross, "tokenAccuracy": {"value": res["tokenAccuracy"], "positions": res["tokenCount"], "note": "subword-level accuracy over SCORED positions only; a different quantity from the span F1 above"},
                   "invalidPredictedTransitions": res["invalidPredictedTransitions"], "maskCheck": {"maxAbsLogitDifference": mask_dev, "claim": "logits of a sentence alone equal its logits inside a padded batch",
                                                                                                           "shortestLen": len(vax[short].ids), "longestLen": len(vax[longest].ids)},
                   "samples": res["records"], "seconds": res["seconds"], "synthetic": True, "note": SYNTH,
                   "provenance": {"source": d.data.get("source"), "vocabSha256": d.data.get("vocabSha256"), "torch": torch.__version__, "model": "Embedding -> BiGRU (packed) -> Linear"}}
        summary["checkpoint"] = CP.persist(ctx, "nlp", signature, res["trainingState"],
            {"architecture": {"vocab": tok.get_vocab_size(), "n_labels": len(tags), "pad_id": pad_id, "emb": cfg.embedding, "hidden": cfg.hidden},
             "tokenizerJson": tok.to_str(), "tokenizerConfig": o["cfg"].model_dump(exclude={"fitted_model_id"}),
             "labels": tags, "maxLength": d.data["contract"]["maxLength"], "labelPolicy": cfg.label_policy, "ignoreIndex": cfg.loss_ignore_index,
             "readout": "first subword of complete words, original character offsets", "evalMode": cfg.eval_mode}, d.data.get("source"), cfg.resume_model_id, {"text": res["records"][0]["text"]})
        return {"report": value(NLP_REPORT, {"metrics": {"microF1": sm["micro"]["f1"], "macroF1": sm["macro"]["f1"]}, "nParams": res["nParams"]})}, summary

    def explain(self, cfg, inputs, outputs):
        return {"equation": "logits = Linear(BiGRU(Embedding(ids)));  loss = cross_entropy(logits, labels, ignore_index)", "rule": W.POLICY_TEXT[cfg.label_policy] + ". Entities are scored per type (precision, recall, F1) the way seqeval does.",
                "note": "Padding is removed from the recurrent state by packing; the run records the measured difference between a sentence alone and in a padded batch."}
