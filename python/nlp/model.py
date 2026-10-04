"""A small token tagger: embedding -> bidirectional GRU over the PACKED (unpadded) sequence -> linear. Padding never reaches the recurrent state, so a sentence gets the same logits alone or in a padded batch."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from tokenizers import Tokenizer

from .iob import prf, validate_iob2
from .subword import Example, batch_tensors, first_subword_positions


class Tagger(nn.Module):
    def __init__(self, vocab: int, n_labels: int, pad_id: int, emb: int = 32, hidden: int = 32):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=pad_id)
        self.rnn = nn.GRU(emb, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, n_labels)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(1).cpu()
        x = nn.utils.rnn.pack_padded_sequence(self.emb(ids), lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(x)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=ids.shape[1])
        return self.out(h)


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    perm = np.random.default_rng(seed).permutation(n).tolist()
    nv = max(1, int(round(n * val_fraction)))
    return sorted(perm[nv:]), sorted(perm[:nv])


def predict_words(model: Tagger, exs: list[Example], id2tag: list[str], pad_id: int, ignore_index: int) -> tuple[list[list[str]], list[list[str]], list[torch.Tensor]]:
    """Gold and predicted WORD-level tag sequences. The word prediction is read at the word's first subword (declared)."""
    model.eval()
    ids, mask, _ = batch_tensors(exs, pad_id, ignore_index)
    with torch.no_grad():
        logits = model(ids, mask)
    gold, pred = [], []
    for i, e in enumerate(exs):
        first = first_subword_positions(e)
        gold.append([w["tag"] for w in e.words])
        pred.append([id2tag[int(logits[i, first[w["wordId"]]].argmax())] for w in e.words])
    return gold, pred, [logits[i, :len(e.ids)] for i, e in enumerate(exs)]


def train_and_eval(exs: list[Example], tok: Tokenizer, id2tag: list[str], *, epochs: int, lr: float, hidden: int, emb: int, batch_size: int, seed: int, ignore_index: int,
                   tr: list[int], va: list[int], n_inspect: int, scheme_mode: str) -> dict[str, Any]:
    pad_id = tok.token_to_id("[PAD]")
    torch.manual_seed(seed)
    model = Tagger(tok.get_vocab_size(), len(id2tag), pad_id, emb, hidden)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    trx = [exs[i] for i in tr]
    vax = [exs[i] for i in va]
    vids, vmask, vlab = batch_tensors(vax, pad_id, ignore_index)
    curve, t0 = [], time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(trx), generator=g).tolist()
        tot, cnt = 0.0, 0
        for k in range(0, len(perm), batch_size):
            b = [trx[i] for i in perm[k:k + batch_size]]
            ids, mask, lab = batch_tensors(b, pad_id, ignore_index)
            logits = model(ids, mask)
            loss = Fn.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1), ignore_index=ignore_index)
            opt.zero_grad()
            loss.backward()
            opt.step()
            n = int((lab != ignore_index).sum())
            tot, cnt = tot + float(loss.detach()) * n, cnt + n
        model.eval()
        with torch.no_grad():
            vl = Fn.cross_entropy((lg := model(vids, vmask)).reshape(-1, lg.shape[-1]), vlab.reshape(-1), ignore_index=ignore_index)
        curve.append({"epoch": ep + 1, "trainLoss": tot / max(cnt, 1), "valLoss": float(vl)})
    gold, pred, logits = predict_words(model, vax, id2tag, pad_id, ignore_index)
    res = prf(gold, pred, scheme_mode)
    # token-level (subword) accuracy over scored positions only
    tok_ok = tok_n = 0
    for e, lg in zip(vax, logits):
        for p, y in enumerate(e.labels):
            if y != ignore_index:
                tok_n += 1
                tok_ok += int(lg[p].argmax()) == y
    strict = prf(gold, pred, "strict")
    invalid_pred = sum(len(validate_iob2(p)) for p in pred)
    recs = []
    for j, e in enumerate(vax[:n_inspect]):
        recs.append({"id": e.id, "text": e.text, "spans": e.spans, "tokens": e.tokens, "ids": e.ids, "offsets": e.offsets, "wordIds": e.word_ids, "attention": [1] * len(e.ids),
                     "labels": [id2tag[y] if y != ignore_index else None for y in e.labels], "labelIds": e.labels,
                     "predTokens": [id2tag[int(logits[j][p].argmax())] for p in range(len(e.ids))],
                     "words": [{**w, "pred": pr} for w, pr in zip(e.words, pred[j])], "trainIndexInCorpus": va[j]})
    return {"nParams": n_params, "curve": curve, "spanMetrics": res, "strictMetrics": strict, "tokenAccuracy": tok_ok / max(tok_n, 1), "tokenCount": tok_n, "invalidPredictedTransitions": invalid_pred,
            "records": recs, "seconds": time.time() - t0, "model": model, "goldWords": gold, "predWords": pred}
