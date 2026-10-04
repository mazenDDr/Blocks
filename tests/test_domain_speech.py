"""Speech workflow (A57): audio contract and the 32,000-sample teaching signal, frame counts, lengths and masks, CTC against a brute-force reference, decoding and error alignment."""
from __future__ import annotations

import itertools
import json
import random

import jiwer
import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import load_domain_generator
from speech import audio as A
from speech import ctc as CT
from speech import features as FE


@pytest.fixture(scope="module")
def clips(domain_fixtures):
    return A.load_npz(domain_fixtures.AUDIO_NPZ, None)


# ------------------------------------------------------------------------------------------------ contract
def test_teaching_signal_is_exactly_32000_samples_per_channel_and_time_follows_the_index():
    t = A.teaching_signal()
    assert (t.sample_rate, t.channels, t.samples_per_channel) == (16000, 1, 32000) and t.duration == 2.0
    assert A.contract(t)["totalSamples"] == 32000 and tuple(t.wave.shape) == (1, 32000)
    assert A.time_of_sample(0, 16000) == 0.0 and A.time_of_sample(16000, 16000) == 1.0 and A.time_of_sample(31999, 16000) == pytest.approx(2.0 - 1 / 16000)
    assert A.sample_of_time(2.0, 16000) == 32000
    stereo = A.Clip("s", torch.zeros(2, 32000), 16000)
    assert stereo.samples_per_channel == 32000 and A.contract(stereo)["totalSamples"] == 64000  # per CHANNEL, not total
    assert A.teaching_signal(1.5).samples_per_channel == 24000 and A.teaching_signal(2.0, 8000).samples_per_channel == 16000
    assert A.TEACHING_SAMPLES_PER_CHANNEL == 32000


def test_audio_fixture_is_deterministic_labelled_and_consistent(tmp_path, clips):
    gen = load_domain_generator()
    a, b = gen.write_audio(tmp_path / "a.npz", n=10), gen.write_audio(tmp_path / "b.npz", n=10)
    assert a.read_bytes() == b.read_bytes()
    cl, spec = clips
    assert spec["synthetic"] is True and spec["sampleRate"] == 16000
    assert cl[0].samples_per_channel == 32000 and {c.samples_per_channel for c in cl} == {16000, 24000, 32000}
    for c in cl:
        assert "".join(s["symbol"] for s in c.segments) == c.text and c.segments[-1]["end"] <= c.samples_per_channel
        assert not (" " in (c.text[0], c.text[-1])) and "  " not in c.text


def test_the_true_tone_is_where_the_fixture_says_it_is(clips):
    cl, spec = clips
    c = cl[3]
    seg = c.segments[0]
    x = c.wave[0, seg["start"]:seg["end"]].numpy()
    f = np.fft.rfftfreq(len(x), 1 / 16000)[np.abs(np.fft.rfft(x * np.hanning(len(x)))).argmax()]
    assert abs(f - spec["tones"][seg["symbol"]]) < 25


def test_resample_changes_samples_not_duration(clips):
    c = clips[0][0]
    r = A.resample(c, 8000)
    assert r.sample_rate == 8000 and r.samples_per_channel == 16000 and r.duration == c.duration
    assert r.segments[0]["start"] == round(c.segments[0]["start"] / 2)


# ------------------------------------------------------------------------------------------------ framing
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("n_fft,hop", [(400, 160), (512, 128), (256, 256), (64, 7), (401, 160), (65, 7)])
@pytest.mark.parametrize("n", [32000, 24001, 16000, 1000, 400])
def test_frame_count_formula_equals_the_real_stft(center, n_fft, hop, n):
    x = torch.randn(n)
    kw = {"pad_mode": "reflect"} if center else {}
    if not center and n < n_fft:
        assert FE.frame_count(n, n_fft, hop, center) == 0  # torch.stft refuses such input; the workbench reports E_AUDIO_TOO_SHORT before running
        return
    if center and n_fft // 2 >= n:
        pytest.skip("reflect padding needs more samples than the pad")
    s = torch.stft(x, n_fft, hop, window=torch.hann_window(n_fft), center=center, return_complex=True, **kw)
    assert s.shape[-1] == FE.frame_count(n, n_fft, hop, center)


def test_frame_count_with_constant_padding_and_too_short_signals():
    x = torch.randn(1000)
    s = torch.stft(x, 400, 160, window=torch.hann_window(400), center=True, pad_mode="constant", return_complex=True)
    assert s.shape[-1] == FE.frame_count(1000, 400, 160, True) == 7
    assert FE.frame_count(399, 400, 160, False) == 0 and FE.frame_count(400, 400, 160, False) == 1 and FE.frame_count(559, 400, 160, False) == 1 and FE.frame_count(560, 400, 160, False) == 2


def test_duration_alone_does_not_determine_the_frame_count():
    """Same duration, different hop / padding: different frame counts (VISION: 'must not be guessed from duration alone')."""
    n = 32000
    assert {FE.frame_count(n, 400, 160, True), FE.frame_count(n, 400, 160, False), FE.frame_count(n, 512, 256, True), FE.frame_count(n, 512, 128, False)} == {201, 198, 126, 247}


def test_mel_features_equal_torchaudio_and_have_the_formula_frames(clips):
    import torchaudio.transforms as AT

    cl, _ = clips
    cfg = FE.FeatureConfig()
    c = cl[0]
    f = FE.features_for(c.wave, cfg, 16000)
    assert f.shape == (201, 40)
    ref = AT.MelSpectrogram(16000, n_fft=400, win_length=400, hop_length=160, n_mels=40, power=2.0, center=True, pad_mode="reflect", norm=None, mel_scale="htk")(c.wave)[0]
    assert torch.allclose(f, torch.log(ref.T + 1e-6), atol=1e-5)
    c2 = FE.FeatureConfig(center=False, n_fft=512, win_length=400, hop_length=256, pad_mode="constant")
    assert FE.features_for(c.wave, c2, 16000).shape[0] == FE.frame_count(32000, 512, 256, False) == 124


def test_window_and_hop_are_in_samples_so_the_frame_time_depends_on_the_sample_rate():
    assert FE.frame_time(0, 160, 16000, True, 400) == (-0.0125, 0.0) and FE.frame_time(100, 160, 16000, True, 400)[1] == pytest.approx(1.0)
    assert FE.frame_time(100, 160, 8000, True, 400)[1] == pytest.approx(2.0)
    assert FE.check_config(FE.FeatureConfig(win_length=500), 16000)[0][0] == "E_AUDIO_FRAME_CONFIG"
    assert FE.check_config(FE.FeatureConfig(f_max=9000.0), 16000)[0][0] == "E_AUDIO_FRAME_CONFIG"


# ------------------------------------------------------------------------------------------------ lengths and masks
def test_padded_batch_lengths_and_masks(clips):
    cl, _ = clips
    cfg = FE.FeatureConfig()
    sub = [cl[i] for i in (0, 5, 6, 1)]
    feats = [FE.features_for(c.wave, cfg, 16000) for c in sub]
    x, lengths, mask = FE.pad_features(feats)
    assert lengths.tolist() == [FE.frame_count(c.samples_per_channel, 400, 160, True) for c in sub] and x.shape == (4, int(lengths.max()), 40)
    for i, f in enumerate(feats):
        n = f.shape[0]
        assert torch.equal(x[i, :n], f) and (x[i, n:] == 0).all() and mask[i].sum() == n and mask[i, :n].all()
    assert len({int(v) for v in lengths}) > 1  # the clips really differ in length


def test_conv_output_length_formula_equals_the_real_convolution():
    for T in (1, 2, 3, 10, 99, 201):
        conv = nn.Conv1d(4, 4, CT.KERNEL, stride=CT.STRIDE, padding=CT.PADDING)
        assert conv(torch.zeros(1, 4, T)).shape[-1] == int(FE.conv_out_len(torch.tensor(T), CT.KERNEL, CT.STRIDE, CT.PADDING))


def test_ctc_model_is_independent_of_padding(clips):
    cl, _ = clips
    cfg = FE.FeatureConfig()
    feats = [FE.features_for(cl[i].wave, cfg, 16000) for i in (0, 5, 6)]
    torch.manual_seed(0)
    m = CT.CTCModel(40, 6, 16).eval()
    x, lengths, _ = FE.pad_features(feats)
    with torch.no_grad():
        lp, ol = m(x, lengths)
        for i, f in enumerate(feats):
            a, al = m(f[None], torch.tensor([f.shape[0]]))
            assert int(al[0]) == int(ol[i])
            assert torch.allclose(lp[i, :int(ol[i])], a[0, :int(al[0])], atol=1e-4)


# ------------------------------------------------------------------------------------------------ CTC
def _brute_force_ctc(logp: torch.Tensor, target: list[int], blank: int = 0) -> float:
    """-log sum over every frame path whose collapse (merge repeats, drop blanks) equals the target."""
    T, V = logp.shape
    total = 0.0
    for path in itertools.product(range(V), repeat=T):
        out, prev = [], blank
        for p in path:
            if p != prev and p != blank:
                out.append(p)
            prev = p
        if out == target:
            total += float(np.exp(sum(float(logp[t, p]) for t, p in enumerate(path))))
    return -float(np.log(total))


@pytest.mark.parametrize("target", [[1], [1, 2], [2, 2], [1, 2, 1]])
def test_ctc_loss_equals_brute_force_over_all_alignments(target):
    torch.manual_seed(0)
    T, V = 5, 3
    logp = torch.randn(T, V).log_softmax(-1)
    loss = nn.CTCLoss(blank=0, reduction="sum")(logp[:, None], torch.tensor(target)[None], torch.tensor([T]), torch.tensor([len(target)]))
    assert float(loss) == pytest.approx(_brute_force_ctc(logp, target), rel=1e-5)
    mean = nn.CTCLoss(blank=0, reduction="mean")(logp[:, None], torch.tensor(target)[None], torch.tensor([T]), torch.tensor([len(target)]))
    assert float(mean) == pytest.approx(float(loss) / len(target), rel=1e-5)  # 'mean' divides by the TARGET length


def test_alignment_constraint_and_infeasible_error(clips):
    assert CT.min_frames_needed([1, 2, 3]) == 3 and CT.min_frames_needed([1, 1, 2, 2]) == 6 and CT.min_frames_needed([]) == 0
    logp = torch.randn(2, 1, 3).log_softmax(-1)
    inf = nn.CTCLoss(blank=0, reduction="sum")(logp, torch.tensor([[1, 1]]), torch.tensor([2]), torch.tensor([2]))
    assert torch.isinf(inf)  # [1, 1] needs 3 frames (a blank between the repeats)
    feats = [torch.randn(3, 40), torch.randn(3, 40)]
    with pytest.raises(ValueError, match="E_CTC_ALIGNMENT_INFEASIBLE"):
        CT.train_and_eval(feats, ["abababab", "ab"], ["a", "b"], epochs=1, lr=1e-3, hidden=8, batch_size=2, seed=0, val_fraction=0.5, n_inspect=1, hop_seconds=0.01)


def test_greedy_decode_collapses_repeats_and_drops_blanks_with_frame_indices():
    # frames: blank a a blank a b b blank  -> a a b  (the blank between the two a's keeps them apart)
    path = [0, 1, 1, 0, 1, 2, 2, 0]
    logp = torch.full((8, 3), -9.0)
    for t, p in enumerate(path):
        logp[t, p] = -0.01
    ids, frames = CT.greedy_decode(logp, 8)
    assert ids == [1, 1, 2] and frames == [1, 4, 5]
    assert CT.greedy_decode(logp, 3)[0] == [1]  # only the first 3 frames are real


# ------------------------------------------------------------------------------------------------ error alignment
def test_edit_alignment_shows_substitution_insertion_deletion():
    a = CT.edit_align(list("kitten"), list("sitting"))
    assert (a["substitutions"], a["deletions"], a["insertions"], a["distance"]) == (2, 0, 1, 3) and a["rate"] == pytest.approx(3 / 6)
    assert [o["op"] for o in a["ops"]] == ["S", "C", "C", "C", "S", "C", "I"]
    d = CT.edit_align(list("abc"), list("ac"))
    assert [(o["op"], o["ref"], o["hyp"]) for o in d["ops"]] == [("C", "a", "a"), ("D", "b", None), ("C", "c", "c")]
    assert CT.edit_align([], list("ab"))["rate"] is None


def test_cer_wer_match_jiwer_on_random_strings():
    rng = random.Random(0)
    for _ in range(80):
        ref = " ".join("".join(rng.choice("abcd") for _ in range(rng.randint(1, 4))) for _ in range(rng.randint(1, 5)))
        hyp = "".join(rng.choice("abcd ") for _ in range(rng.randint(0, 14))).strip()
        hyp = " ".join(hyp.split())
        if not hyp:
            continue
        o_w, o_c = jiwer.process_words(ref, hyp), jiwer.process_characters(ref, hyp)
        a = CT.cer_wer(ref, hyp)
        # the minimal edit COUNT is unique; how it splits into S/D/I can differ between equally good alignments, so compare the rate and check our own ops are a valid script
        assert a["wer"]["rate"] == pytest.approx(o_w.wer) and a["cer"]["rate"] == pytest.approx(o_c.cer)
        for key, toks in (("cer", (list(ref), list(hyp))), ("wer", (ref.split(), hyp.split()))):
            al = a[key]
            assert al["substitutions"] + al["deletions"] + al["insertions"] == al["distance"]
            assert [o["ref"] for o in al["ops"] if o["ref"] is not None] == toks[0] and [o["hyp"] for o in al["ops"] if o["hyp"] is not None] == toks[1]


def test_corpus_rate_is_total_edits_over_total_reference_not_a_mean_of_rates():
    r = CT.corpus_rates([("ab", "ab"), ("abcd", "")])
    assert r["cer"]["rate"] == pytest.approx(4 / 6) and r["cer"]["deletions"] == 4 and r["wer"]["rate"] == pytest.approx(1 / 2)


# ------------------------------------------------------------------------------------------------ training
def test_ctc_recogniser_learns_the_tone_sequences(clips):
    cl, spec = clips
    cfg = FE.FeatureConfig()
    sub = cl[:30]
    feats = [FE.features_for(c.wave, cfg, 16000) for c in sub]
    res = CT.train_and_eval(feats, [c.text for c in sub], spec["alphabet"], epochs=60, lr=5e-3, hidden=32, batch_size=4, seed=0, val_fraction=0.25, n_inspect=2, hop_seconds=0.01)
    assert res["curve"][-1]["trainLoss"] < 0.5 * res["curve"][0]["trainLoss"]
    assert res["rates"]["cer"]["rate"] < 1.0 and len(res["records"]) == 2
    r = res["records"][0]
    assert r["outFrames"] == int(FE.conv_out_len(r["frames"], 3, 2, 1)) and len(r["tokenFrames"]) == len(r["hyp"]) and r["minFramesNeeded"] <= r["outFrames"]


def test_resampling_nondivisible_lengths_uses_native_ceiling():
    c = A.Clip("odd", torch.zeros(1, 1001), 16000)
    r = A.resample(c, 8000)
    assert r.samples_per_channel == 501


def test_invalid_mel_frequency_range_is_refused():
    assert FE.check_config(FE.FeatureConfig(f_min=-1), 16000)[0][0] == "E_AUDIO_FRAME_CONFIG"
    assert FE.check_config(FE.FeatureConfig(f_min=8000), 16000)[0][0] == "E_AUDIO_FRAME_CONFIG"
