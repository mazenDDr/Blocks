"""IOB2 validation and span-level evaluation following the `seqeval` conventions (the default conlleval-compatible mode and the strict IOB2 mode).

An entity is (type, start, end) over word positions, end exclusive here (seqeval's own get_entities uses an inclusive end; `seqeval_entities` converts when comparing).
Default mode (conlleval): `I-X` after `O`, or after a different type, STARTS an entity, so a prediction stream is never rejected, only scored.
Strict IOB2 mode: an entity must start with `B-X`; stray `I-X` tags are not entities."""
from __future__ import annotations

from typing import Iterable

Entity = tuple[str, int, int]


def split_tag(tag: str) -> tuple[str, str]:
    if tag == "O":
        return "O", ""
    if len(tag) > 2 and tag[1] == "-" and tag[0] in "BI":
        return tag[0], tag[2:]
    return "?", tag


def validate_iob2(tags: list[str]) -> list[dict]:
    """Violations of the IOB2 grammar: unknown tag shapes, and `I-X` that does not follow `B-X` or `I-X`."""
    out, prev = [], "O"
    for i, t in enumerate(tags):
        p, ty = split_tag(t)
        if p == "?":
            out.append({"position": i, "tag": t, "code": "E_NLP_TAG_SHAPE", "message": f"'{t}' is not O, B-<type> or I-<type>"})
        elif p == "I":
            pp, pty = split_tag(prev)
            if pp not in ("B", "I") or pty != ty:
                out.append({"position": i, "tag": t, "code": "E_NLP_IOB2_INVALID", "message": f"'{t}' follows '{prev}' (an entity must start with B-{ty})"})
        prev = t
    return out


def entities(tags: Iterable[str], mode: str = "default") -> list[Entity]:
    tags = list(tags)
    out: list[Entity] = []
    cur: list | None = None
    prev_p, prev_t = "O", ""
    for i, tag in enumerate(tags):
        p, ty = split_tag(tag)
        if p == "?":  # unknown shape behaves like O
            p, ty = "O", ""
        if mode == "strict":
            if p == "B":
                if cur:
                    out.append((cur[0], cur[1], i))
                cur = [ty, i]
            elif p == "I" and cur and cur[0] == ty:
                pass
            else:
                if cur:
                    out.append((cur[0], cur[1], i))
                cur = None
        else:
            end = prev_p in "BI" and prev_p != "O" and (p in ("B", "O") or (p == "I" and ty != prev_t))
            start = p == "B" or (p == "I" and (prev_p == "O" or ty != prev_t))
            if end and cur:
                out.append((cur[0], cur[1], i))
                cur = None
            if start:
                cur = [ty, i]
        prev_p, prev_t = p, ty
    if cur:
        out.append((cur[0], cur[1], len(tags)))
    return out


def prf(true_seqs: list[list[str]], pred_seqs: list[list[str]], mode: str = "default") -> dict:
    """Entity-level precision / recall / F1 per type, micro and macro averages (the numbers `seqeval.metrics.classification_report` prints)."""
    if len(true_seqs) != len(pred_seqs) or any(len(a) != len(b) for a, b in zip(true_seqs, pred_seqs)):
        raise ValueError("gold and predicted tag sequences must have the same length per sentence")
    types: dict[str, list[int]] = {}
    for si, (a, b) in enumerate(zip(true_seqs, pred_seqs)):
        ta, pb = {(si, *e) for e in entities(a, mode)}, {(si, *e) for e in entities(b, mode)}
        for e in ta | pb:
            row = types.setdefault(e[1], [0, 0, 0])  # tp, n_pred, n_true
            row[0] += e in ta and e in pb
            row[1] += e in pb
            row[2] += e in ta

    def f(tp, npred, ntrue):
        pr = tp / npred if npred else 0.0
        rc = tp / ntrue if ntrue else 0.0
        return {"precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0, "support": ntrue, "predicted": npred, "correct": tp}

    per = {t: f(*v) for t, v in sorted(types.items())}
    tp, npred, ntrue = (sum(v[i] for v in types.values()) for i in range(3))
    micro = f(tp, npred, ntrue)
    macro = {k: (sum(p[k] for p in per.values()) / len(per) if per else 0.0) for k in ("precision", "recall", "f1")}
    return {"perType": per, "micro": micro, "macro": {**macro, "support": ntrue}, "mode": mode}
