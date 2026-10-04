"""Subword tokenization (Hugging Face `tokenizers`, trained offline on the fixture), source character offsets, word-level IOB2 labels from character spans,
and the declared policy that turns word labels into subword labels."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers

from .iob import split_tag, validate_iob2

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
POLICIES = ("first_subword", "all_subwords")
POLICY_TEXT = {
    "first_subword": "the first subword of each word carries the word's label; continuation subwords get the ignore index and are not scored",
    "all_subwords": "every subword is labelled: the first with the word's label, continuations with its I- form (B-X becomes I-X); the word prediction is still read at the first subword",
}


class NlpError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def train_tokenizer(texts: list[str], vocab_size: int, lowercase: bool = True) -> Tokenizer:
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase() if lowercase else normalizers.Sequence([])
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(texts, trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=SPECIAL, continuing_subword_prefix="##"))
    cls, sep = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", cls), ("[SEP]", sep)])
    return tok


def tag_list(types: list[str]) -> list[str]:
    return ["O"] + [f"{p}-{t}" for t in sorted(types) for p in "BI"]


@dataclass
class Example:
    id: str
    text: str
    spans: list[dict]
    tokens: list[str]
    ids: list[int]
    offsets: list[tuple[int, int]]
    word_ids: list[int | None]
    words: list[dict]  # {"start","end","text","tag"}: word-level units (pre-tokenizer words) kept after truncation
    labels: list[int]  # aligned subword labels (ignore index where not scored)
    truncated_words: int = 0
    violations: list[dict] = field(default_factory=list)


def word_tags(words: list[tuple[int, int]], spans: list[dict], ex_id: str) -> list[str]:
    """Word-level IOB2 tags from character spans. A span must start and end on word boundaries, else the label would be ambiguous."""
    tags = ["O"] * len(words)
    for sp in spans:
        inside = [i for i, (s, e) in enumerate(words) if s >= sp["start"] and e <= sp["end"] and e > s]
        starts = {s for s, _ in words}
        ends = {e for _, e in words}
        if not inside or sp["start"] not in starts or sp["end"] not in ends:
            raise NlpError("E_NLP_SPAN_MISALIGNED", f"{ex_id}: span [{sp['start']}, {sp['end']}) ({sp['label']}) does not start and end on word boundaries of the pre-tokenizer")
        for k, i in enumerate(inside):
            if tags[i] != "O":
                raise NlpError("E_NLP_SPAN_OVERLAP", f"{ex_id}: overlapping entity spans at word {i}")
            tags[i] = f"{'B' if k == 0 else 'I'}-{sp['label']}"
    return tags


def encode_example(tok: Tokenizer, rec: dict, tag2id: dict[str, int], max_length: int, policy: str, ignore_index: int) -> Example:
    enc = tok.encode(rec["text"])
    n_tok = len(enc.ids)
    keep = min(n_tok, max_length)
    if keep < n_tok:  # keep [SEP] last
        keep = max_length
    ids, toks, offs, wids, special = list(enc.ids), list(enc.tokens), list(enc.offsets), list(enc.word_ids), list(enc.special_tokens_mask)
    # whole-sentence word list (for validating spans) before truncation
    spans_by_word: dict[int, list[int]] = {}
    for i, w in enumerate(wids):
        if w is not None:
            spans_by_word.setdefault(w, []).append(i)
    word_ranges = {w: (min(offs[i][0] for i in ix), max(offs[i][1] for i in ix)) for w, ix in spans_by_word.items()}
    order = sorted(word_ranges)
    tags_all = word_tags([word_ranges[w] for w in order], rec["spans"], rec["id"])
    tag_of = dict(zip(order, tags_all))
    truncated = n_tok > max_length
    if truncated:
        ids, toks, offs, wids, special = ids[:max_length - 1] + [ids[-1]], toks[:max_length - 1] + [toks[-1]], offs[:max_length - 1] + [offs[-1]], wids[:max_length - 1] + [None], special[:max_length - 1] + [1]
    kept_words = sorted({w for w in wids if w is not None})
    # a word cut in the middle by truncation is dropped from evaluation too
    content_end = max_length - 1 if truncated else len(ids)
    complete = {w for w in kept_words if all(i < content_end for i in spans_by_word[w])}
    labels, seen = [], set()
    for i, w in enumerate(wids):
        if w is None or special[i] or w not in complete:
            labels.append(ignore_index)
            continue
        t = tag_of[w]
        if w not in seen:
            seen.add(w)
            labels.append(tag2id[t])
        elif policy == "all_subwords":
            p, ty = split_tag(t)
            labels.append(tag2id[f"I-{ty}" if p == "B" else t])
        else:
            labels.append(ignore_index)
    words = [{"start": word_ranges[w][0], "end": word_ranges[w][1], "text": rec["text"][word_ranges[w][0]:word_ranges[w][1]], "tag": tag_of[w], "wordId": w} for w in sorted(complete)]
    return Example(rec["id"], rec["text"], rec["spans"], toks, ids, offs, wids, words, labels, len(order) - len(complete), validate_iob2([x["tag"] for x in words]))


def batch_tensors(exs: list[Example], pad_id: int, ignore_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = max(len(e.ids) for e in exs)
    ids = torch.full((len(exs), T), pad_id, dtype=torch.long)
    mask = torch.zeros(len(exs), T, dtype=torch.long)
    lab = torch.full((len(exs), T), ignore_index, dtype=torch.long)
    for i, e in enumerate(exs):
        n = len(e.ids)
        ids[i, :n], mask[i, :n], lab[i, :n] = torch.tensor(e.ids), 1, torch.tensor(e.labels)
    return ids, mask, lab


def first_subword_positions(e: Example) -> dict[int, int]:
    """word id -> position of its first subword."""
    out: dict[int, int] = {}
    for i, w in enumerate(e.word_ids):
        if w is not None and w not in out:
            out[w] = i
    return out
