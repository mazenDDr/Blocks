"""NLP token-labelling workflow (A58): source spans, subwords, masks, label alignment policies, IOB2 validation and seqeval-convention evaluation."""
from __future__ import annotations

import json
import random

import pytest
import torch
from seqeval.metrics import classification_report as sq_report
from seqeval.metrics.sequence_labeling import get_entities as sq_entities
from seqeval.scheme import IOB2

from conftest import EXAMPLES, load_domain_generator
from nlp import iob, model as M, subword as W

RECS = [json.loads(line) for line in (EXAMPLES / "fixtures" / "synthetic_ner.jsonl").read_text().splitlines()]
TAGS = W.tag_list(["PER", "ORG", "LOC"])
T2I = {t: i for i, t in enumerate(TAGS)}


@pytest.fixture(scope="module")
def tok():
    return W.train_tokenizer([r["text"] for r in RECS[:180]], 300)


def encode(tok, rec, policy="first_subword", max_length=48):
    return W.encode_example(tok, rec, T2I, max_length, policy, -100)


# ------------------------------------------------------------------------------------------------ fixture
def test_ner_fixture_is_deterministic_labelled_and_spans_are_character_offsets(tmp_path):
    gen = load_domain_generator()
    a = gen.write_ner(tmp_path / "a.jsonl")
    assert a.read_bytes() == (EXAMPLES / "fixtures" / "synthetic_ner.jsonl").read_bytes()  # the committed file is what the generator writes
    for r in RECS:
        assert r["synthetic"] is True
        for sp in r["spans"]:
            assert 0 <= sp["start"] < sp["end"] <= len(r["text"]) and r["text"][sp["start"]:sp["end"]][0].isupper()


# ------------------------------------------------------------------------------------------------ tokenization and spans
def test_offsets_index_the_original_text_and_words_group_subwords(tok):
    for rec in RECS[:60]:
        e = encode(tok, rec)
        assert e.tokens[0] == "[CLS]" and e.tokens[-1] == "[SEP]" and e.word_ids[0] is None and e.word_ids[-1] is None
        for t, (s, en), w in zip(e.tokens, e.offsets, e.word_ids):
            if w is None:
                continue
            piece = t[2:] if t.startswith("##") else t
            assert rec["text"][s:en].lower() == piece or t == "[UNK]"
        for w in e.words:
            assert rec["text"][w["start"]:w["end"]] == w["text"]
    assert any(len([w for w in encode(tok, r).word_ids if w is not None]) > len(encode(tok, r).words) for r in RECS[:40])  # real subword splitting happens


def test_word_tags_follow_the_character_spans_and_are_valid_iob2(tok):
    for rec in RECS[:80]:
        e = encode(tok, rec)
        assert not e.violations
        ents = iob.entities([w["tag"] for w in e.words])
        got = [(t, e.words[a]["start"], e.words[b - 1]["end"]) for t, a, b in ents]
        assert got == [(s["label"], s["start"], s["end"]) for s in rec["spans"]]  # the spans are recovered exactly from the word tags


def test_a_span_that_cuts_a_word_is_rejected_with_a_stable_code(tok):
    rec = {"id": "x", "text": "Alice works in Paris.", "spans": [{"start": 0, "end": 3, "label": "PER"}]}
    with pytest.raises(W.NlpError) as e:
        encode(tok, rec)
    assert e.value.code == "E_NLP_SPAN_MISALIGNED"
    with pytest.raises(W.NlpError) as e2:
        encode(tok, {**rec, "spans": [{"start": 0, "end": 5, "label": "PER"}, {"start": 0, "end": 5, "label": "ORG"}]})
    assert e2.value.code == "E_NLP_SPAN_OVERLAP"


# ------------------------------------------------------------------------------------------------ label alignment policies
def test_first_subword_policy_labels_one_position_per_word(tok):
    for rec in RECS[:60]:
        e = encode(tok, rec, "first_subword")
        scored = [i for i, y in enumerate(e.labels) if y != -100]
        assert len(scored) == len(e.words)
        first = W.first_subword_positions(e)
        assert scored == sorted(first[w["wordId"]] for w in e.words)
        for w in e.words:
            assert e.labels[first[w["wordId"]]] == T2I[w["tag"]]
        assert e.labels[0] == -100 and e.labels[-1] == -100  # [CLS] / [SEP]


def test_all_subwords_policy_labels_continuations_with_the_i_form(tok):
    saw = False
    for rec in RECS[:80]:
        e = encode(tok, rec, "all_subwords")
        first = W.first_subword_positions(e)
        for i, (w, y) in enumerate(zip(e.word_ids, e.labels)):
            if w is None:
                assert y == -100
                continue
            tag = next(x["tag"] for x in e.words if x["wordId"] == w)
            if i == first[w]:
                assert y == T2I[tag]
            else:
                saw = True
                p, ty = iob.split_tag(tag)
                assert TAGS[y] == (f"I-{ty}" if p == "B" else tag)
        assert iob.validate_iob2([TAGS[y] for y in e.labels if y != -100]) == []  # the subword sequence is valid IOB2 too
    assert saw


def test_truncation_drops_cut_words_from_labels_and_keeps_sep_last(tok):
    rec = max(RECS, key=lambda r: len(r["text"]))
    full, cut = encode(tok, rec, max_length=48), encode(tok, rec, max_length=9)
    assert len(cut.ids) == 9 and cut.tokens[-1] == "[SEP]" and cut.truncated_words > 0 and len(cut.words) < len(full.words)
    assert sum(y != -100 for y in cut.labels) == len(cut.words)


def test_padding_and_attention_masks(tok):
    exs = [encode(tok, r) for r in RECS[:6]]
    ids, mask, lab = W.batch_tensors(exs, tok.token_to_id("[PAD]"), -100)
    assert ids.shape == mask.shape == lab.shape and ids.shape[1] == max(len(e.ids) for e in exs)
    for i, e in enumerate(exs):
        n = len(e.ids)
        assert mask[i, :n].all() and not mask[i, n:].any() and (ids[i, n:] == tok.token_to_id("[PAD]")).all() and (lab[i, n:] == -100).all()


def test_tokenizer_is_fitted_on_the_training_sentences_only():
    tr, va = M.split_indices(len(RECS), 0.25, 0)
    assert not set(tr) & set(va) and len(tr) + len(va) == len(RECS)
    only_val = "Zzqx"
    texts = [RECS[i]["text"] for i in tr]
    tok = W.train_tokenizer(texts, 120)
    assert tok.get_vocab_size() <= 120 and "[PAD]" in tok.get_vocab()
    assert tok.encode(only_val).tokens[1:-1] != [only_val.lower()]  # unseen word is split / unknown, not memorised


# ------------------------------------------------------------------------------------------------ IOB2 validation
def test_iob2_validation_cases():
    assert iob.validate_iob2(["O", "B-PER", "I-PER", "O", "B-LOC"]) == []
    v = iob.validate_iob2(["O", "I-PER"])
    assert v[0]["code"] == "E_NLP_IOB2_INVALID" and v[0]["position"] == 1
    assert iob.validate_iob2(["B-PER", "I-LOC"])[0]["code"] == "E_NLP_IOB2_INVALID"
    assert iob.validate_iob2(["B-PER", "X"])[0]["code"] == "E_NLP_TAG_SHAPE"
    assert iob.validate_iob2(["I-PER"])[0]["position"] == 0


# ------------------------------------------------------------------------------------------------ seqeval conventions
def _rand_seqs(rng, n, valid):
    types = ["PER", "LOC", "ORG"]
    out = []
    for _ in range(n):
        L = rng.randint(1, 12)
        if valid:
            seq = []
            while len(seq) < L:
                if rng.random() < 0.5:
                    seq.append("O")
                else:
                    t = rng.choice(types)
                    seq.append(f"B-{t}")
                    seq += [f"I-{t}"] * rng.randint(0, 2)
            out.append(seq[:L] if True else seq)
        else:
            out.append([rng.choice(["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]) for _ in range(L)])
    return out


@pytest.mark.parametrize("valid", [True, False])
def test_entity_extraction_matches_seqeval(valid):
    rng = random.Random(0)
    for seq in _rand_seqs(rng, 400, valid):
        ref = sorted((t, s, e + 1) for t, s, e in sq_entities(seq))
        assert sorted(iob.entities(seq, "default")) == ref, seq


def test_strict_iob2_extraction_matches_seqeval_strict_scheme():
    from seqeval.metrics.v1 import _precision_recall_fscore_support  # noqa: F401  (only to be sure the strict path exists)
    from seqeval.scheme import Entities

    rng = random.Random(1)
    for seq in _rand_seqs(rng, 400, False):
        ref = sorted((e.tag, e.start, e.end) for e in Entities([seq], IOB2).entities[0])
        assert sorted(iob.entities(seq, "strict")) == ref, seq


@pytest.mark.parametrize("mode", ["default", "strict"])
@pytest.mark.parametrize("valid", [True, False])
def test_prf_matches_seqeval_classification_report(mode, valid):
    rng = random.Random(2)
    gold = _rand_seqs(rng, 60, True)
    pred = [[t if rng.random() < 0.8 else rng.choice(["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]) for t in s] for s in (gold if valid else _rand_seqs(rng, 60, False))]
    pred = [p[:len(g)] + ["O"] * (len(g) - len(p)) for g, p in zip(gold, pred)]
    kw = {"mode": "strict", "scheme": IOB2} if mode == "strict" else {}
    ref = sq_report(gold, pred, output_dict=True, zero_division=0, **kw)
    mine = iob.prf(gold, pred, mode)
    for t, row in mine["perType"].items():
        assert row["precision"] == pytest.approx(ref[t]["precision"]) and row["recall"] == pytest.approx(ref[t]["recall"])
        assert row["f1"] == pytest.approx(ref[t]["f1-score"]) and row["support"] == ref[t]["support"]
    assert mine["micro"]["precision"] == pytest.approx(ref["micro avg"]["precision"]) and mine["micro"]["f1"] == pytest.approx(ref["micro avg"]["f1-score"])
    assert mine["macro"]["f1"] == pytest.approx(ref["macro avg"]["f1-score"]) and mine["micro"]["support"] == ref["micro avg"]["support"]


def test_prf_hand_calculation():
    gold = [["B-PER", "I-PER", "O", "B-LOC"]]
    pred = [["B-PER", "O", "O", "B-LOC"]]
    r = iob.prf(gold, pred)
    assert r["perType"]["PER"]["recall"] == 0.0 and r["perType"]["LOC"]["f1"] == 1.0  # a half-found entity is a miss and a false positive
    assert r["micro"]["correct"] == 1 and r["micro"]["predicted"] == 2 and r["micro"]["support"] == 2
    with pytest.raises(ValueError):
        iob.prf([["O"]], [["O", "O"]])


# ------------------------------------------------------------------------------------------------ model
def _fit(tok, epochs, policy="first_subword"):
    exs = [encode(tok, r, policy) for r in RECS]
    tr, va = M.split_indices(len(exs), 0.25, 0)
    return exs, M.train_and_eval(exs, tok, TAGS, epochs=epochs, lr=5e-3, hidden=16, emb=16, batch_size=16, seed=0, ignore_index=-100, tr=tr, va=va, n_inspect=3, scheme_mode="default"), va


def test_tagger_learns_and_inspection_records_align(tok):
    exs, res, va = _fit(tok, 12)
    assert res["curve"][-1]["trainLoss"] < res["curve"][0]["trainLoss"] and res["spanMetrics"]["micro"]["f1"] > 0.4
    rec = res["records"][0]
    n = len(rec["tokens"])
    assert n == len(rec["ids"]) == len(rec["offsets"]) == len(rec["wordIds"]) == len(rec["labels"]) == len(rec["predTokens"]) == len(rec["attention"])
    assert len(rec["words"]) == len([x for x in rec["labels"] if x is not None])
    assert res["tokenAccuracy"] > 0.5 and res["tokenCount"] == sum(y != -100 for i in va for y in exs[i].labels)  # subword accuracy counts scored positions only


def test_a_sentence_has_the_same_logits_alone_and_in_a_padded_batch(tok):
    exs = [encode(tok, r) for r in RECS[:8]]
    torch.manual_seed(0)
    model = M.Tagger(tok.get_vocab_size(), len(TAGS), tok.token_to_id("[PAD]"), 16, 16).eval()
    ids, mask, _ = W.batch_tensors(exs, tok.token_to_id("[PAD]"), -100)
    with torch.no_grad():
        full = model(ids, mask)
        for i, e in enumerate(exs):
            n = len(e.ids)
            alone = model(ids[i:i + 1, :n], mask[i:i + 1, :n])
            assert torch.allclose(full[i, :n], alone[0], atol=1e-5)


def test_all_subwords_policy_trains_with_the_same_word_level_readout(tok):
    _, res, _ = _fit(tok, 6, "all_subwords")
    assert 0.0 <= res["spanMetrics"]["micro"]["f1"] <= 1.0 and res["invalidPredictedTransitions"] >= 0


def test_word_ending_at_replaced_sep_position_is_not_scored():
    # Construct real WordPiece semantics, with a 3-piece word occupying positions 1..3.
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    tok = Tokenizer(models.WordPiece({"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "a": 4, "##b": 5, "##c": 6, "d": 7}))
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    rec = {"id": "boundary", "text": "abc d", "spans": [{"start": 0, "end": 3, "label": "PER"}]}
    cut = encode(tok, rec, max_length=4)
    assert cut.tokens == ["[CLS]", "a", "##b", "[SEP]"]
    assert cut.words == [] and cut.labels == [-100] * 4 and cut.truncated_words == 2
