"""A tiny CTC recogniser (torch.nn.CTCLoss), greedy decoding with frame timestamps, and error alignment (edit-distance backtrace with S/D/I, CER, WER)."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .features import conv_out_len

KERNEL, STRIDE, PADDING = 3, 2, 1  # the only downsampling in the model: time axis 2x


class CTCModel(nn.Module):
    def __init__(self, n_mels: int, vocab_with_blank: int, hidden: int = 48):
        super().__init__()
        self.conv = nn.Conv1d(n_mels, hidden, KERNEL, stride=STRIDE, padding=PADDING)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, vocab_with_blank)

    def out_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return conv_out_len(lengths, KERNEL, STRIDE, PADDING)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B, T, n_mels) padded, lengths (B,) real frames -> log-probs (B, T', V+1), out lengths (B,). Padding is excluded from the recurrent state by packing."""
        h = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        ol = self.out_lengths(lengths)
        pk = nn.utils.rnn.pack_padded_sequence(h, ol.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = self.rnn(pk)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=h.shape[1])
        return Fn.log_softmax(self.out(o), -1), ol


def min_frames_needed(target: list[int]) -> int:
    """A CTC alignment needs one frame per label plus a blank between every pair of identical neighbours."""
    return len(target) + sum(1 for a, b in zip(target, target[1:]) if a == b)


def greedy_decode(logp: torch.Tensor, length: int, blank: int = 0) -> tuple[list[int], list[int]]:
    """Best path: argmax per frame, collapse repeats, drop blanks. Returns (labels, frame index where each label first appears)."""
    path = logp[:length].argmax(-1).tolist()
    out, frames, prev = [], [], blank
    for t, p in enumerate(path):
        if p != prev and p != blank:
            out.append(p)
            frames.append(t)
        prev = p
    return out, frames


def edit_align(ref: list, hyp: list) -> dict[str, Any]:
    """Levenshtein alignment (unit costs) with a backtrace. Ties prefer: match/substitution, then deletion, then insertion."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int64)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1, j] + 1, d[i, j - 1] + 1)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append({"op": "C" if ref[i - 1] == hyp[j - 1] else "S", "ref": ref[i - 1], "hyp": hyp[j - 1]})
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append({"op": "D", "ref": ref[i - 1], "hyp": None})
            i -= 1
        else:
            ops.append({"op": "I", "ref": None, "hyp": hyp[j - 1]})
            j -= 1
    ops.reverse()
    c = {k: sum(1 for o in ops if o["op"] == k) for k in "CSDI"}
    return {"ops": ops, "distance": int(d[n, m]), "substitutions": c["S"], "deletions": c["D"], "insertions": c["I"], "correct": c["C"], "refLength": n,
            "rate": (int(d[n, m]) / n) if n else None}


def cer_wer(ref: str, hyp: str) -> dict[str, Any]:
    """CER over characters (the space counts as a character), WER over whitespace-separated words."""
    return {"cer": edit_align(list(ref), list(hyp)), "wer": edit_align(ref.split(), hyp.split())}


def corpus_rates(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """Corpus-level rates: total edits / total reference length (not the mean of per-utterance rates)."""
    out = {}
    for name, tok in (("cer", list), ("wer", str.split)):
        s = d = i = n = 0
        for ref, hyp in pairs:
            a = edit_align(tok(ref), tok(hyp))
            s, d, i, n = s + a["substitutions"], d + a["deletions"], i + a["insertions"], n + a["refLength"]
        out[name] = {"rate": (s + d + i) / n if n else None, "substitutions": s, "deletions": d, "insertions": i, "refLength": n}
    return out


def train_and_eval(feats: list[torch.Tensor], texts: list[str], alphabet: list[str], *, epochs: int, lr: float, hidden: int, batch_size: int, seed: int, val_fraction: float,
                   n_inspect: int, hop_seconds: float, blank: int = 0, reduction: str = "mean", zero_infinity: bool = False) -> dict[str, Any]:
    from .features import pad_features

    sym2id = {c: i + 1 for i, c in enumerate(alphabet)}
    targets = [[sym2id[c] for c in t] for t in texts]
    n = len(feats)
    perm = np.random.default_rng(seed).permutation(n).tolist()
    nv = max(1, int(round(n * val_fraction)))
    va, tr = sorted(perm[:nv]), sorted(perm[nv:])
    # per-mel-band standardisation fitted on the TRAINING clips only
    cat = torch.cat([feats[i] for i in tr])
    mu, sd = cat.mean(0), cat.std(0).clamp(min=1e-5)
    raw = feats
    feats = [(f - mu) / sd for f in feats]
    torch.manual_seed(seed)
    model = CTCModel(feats[0].shape[1], len(alphabet) + 1, hidden)
    # alignment feasibility is checked BEFORE training: CTC needs T_out >= label count + repeats
    out_len_all = model.out_lengths(torch.tensor([f.shape[0] for f in feats]))
    for i in range(n):
        need = min_frames_needed(targets[i])
        if int(out_len_all[i]) < need:
            raise ValueError(f"E_CTC_ALIGNMENT_INFEASIBLE:clip {i} has {int(out_len_all[i])} output frames but its transcript needs at least {need}")
    ctc = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)

    def loss_of(ix: list[int]):
        x, lens, _ = pad_features([feats[i] for i in ix])
        logp, ol = model(x, lens)
        tg = torch.cat([torch.tensor(targets[i]) for i in ix])
        tl = torch.tensor([len(targets[i]) for i in ix])
        return ctc(logp.transpose(0, 1), tg, ol, tl), logp, ol

    curve, t0 = [], time.time()
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(tr), generator=g).tolist()
        tot = 0.0
        for k in range(0, len(order), batch_size):
            ix = [tr[j] for j in order[k:k + batch_size]]
            loss, _, _ = loss_of(ix)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(ix)
        model.eval()
        with torch.no_grad():
            vl = float(loss_of(va)[0])
        curve.append({"epoch": ep + 1, "trainLoss": tot / len(tr), "valLoss": vl})
    model.eval()
    with torch.no_grad():
        _, logp, ol = loss_of(va)
    id2sym = {v: k for k, v in sym2id.items()}
    pairs, recs = [], []
    for j, i in enumerate(va):
        ids, frames = greedy_decode(logp[j], int(ol[j]), blank)
        hyp = "".join(id2sym[x] for x in ids)
        pairs.append((texts[i], hyp))
        if j < n_inspect:
            lp = logp[j, :int(ol[j])]
            recs.append({"index": i, "ref": texts[i], "hyp": hyp, "frames": int(feats[i].shape[0]), "outFrames": int(ol[j]), "targetLength": len(targets[i]),
                         "minFramesNeeded": min_frames_needed(targets[i]), "tokenFrames": frames,
                         "framePath": lp.argmax(-1).tolist(), "frameProb": [round(float(x), 3) for x in lp.max(-1).values.exp()],
                         "align": cer_wer(texts[i], hyp), "spectrogram": raw[i].numpy()})
    return {"model": model, "curve": curve, "rates": corpus_rates(pairs), "records": recs, "valIdx": va, "trainIdx": tr, "seconds": time.time() - t0,
            "nParams": sum(p.numel() for p in model.parameters()), "outLenFactor": STRIDE, "valPairs": pairs, "alphabet": alphabet,
            "normalization": {"fittedOn": "training clips", "nTrainFrames": int(cat.shape[0])}, "ctc": {"blank": blank, "reduction": reduction, "zeroInfinity": zero_infinity}}
