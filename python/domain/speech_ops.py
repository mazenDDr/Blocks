"""Speech operations of the `domain` graph kind: audio source with the sample-rate contract, resampling, framed/mel features with a verified frame-count formula, and a tiny CTC recogniser."""
from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import torch
from pydantic import Field

from graph_core.registry import register
from graph_core.types import Fix, OpError
from operations._common import StrictConfig
from speech import audio as A
from speech import ctc as CT
from speech import features as FE
from tabular.core import ExecutionError, VType, resolve_path

from . import checkpoints as CP
from .core import AUDIO_BATCH, FEATURE_BATCH, SPEECH_REPORT, DomainOperation, fixture_path, peek_spec, r, sha256_file, value

DEFAULT_NPZ = "examples/data/domain/synthetic_tones.npz"
SYNTH = ("SYNTHETIC fixture (examples/make_domain_fixtures.py): generated tone sequences (a 300 Hz, b 600 Hz, c 1000 Hz, d 1500 Hz, space 2200 Hz), not recorded speech.")


def envelope(x: np.ndarray, bins: int = 800) -> dict[str, Any]:
    n = len(x)
    edges = np.linspace(0, n, min(bins, n) + 1).astype(int)
    lo = [float(x[a:b].min()) for a, b in zip(edges[:-1], edges[1:])]
    hi = [float(x[a:b].max()) for a, b in zip(edges[:-1], edges[1:])]
    return {"samples": n, "bins": len(lo), "min": [round(v, 3) for v in lo], "max": [round(v, 3) for v in hi], "samplesPerBin": n / len(lo)}


def clip_record(c: A.Clip, with_segments: bool = True) -> dict[str, Any]:
    w = c.wave[0].numpy()
    d = {**A.contract(c), "text": c.text, "envelope": envelope(w), "rawHead": [round(float(v), 4) for v in w[:480]], "timeOfLastSample": A.time_of_sample(c.samples_per_channel - 1, c.sample_rate)}
    if with_segments and c.segments:
        d["segments"] = [{"symbol": s["symbol"], "start": s["start"], "end": s["end"], "t0": s["start"] / c.sample_rate, "t1": s["end"] / c.sample_rate} for s in c.segments]
    return d


def _batch_data(clips: list[A.Clip], spec_alphabet: list[str] | None, extra: dict[str, Any]) -> dict[str, Any]:
    lens = [c.samples_per_channel for c in clips]
    return {"contract": {"n": len(clips), "sampleRate": clips[0].sample_rate, "channels": clips[0].channels, "minSamples": min(lens), "maxSamples": max(lens),
                         "durationsSeconds": sorted({round(c.duration, 6) for c in clips}), "alphabet": spec_alphabet}, **extra}


def _info(raw: dict[str, Any] | None, rate: int | None = None) -> dict[str, Any]:
    if not raw:
        return {"sampleRate": rate, "channels": 1, "minSamples": None, "maxSamples": None, "alphabet": None}
    sr = raw["sampleRate"]
    ds = raw["durationsSeconds"]
    return {"sampleRate": sr, "channels": raw["channels"], "minSamples": int(round(min(ds) * sr)), "maxSamples": int(round(max(ds) * sr)), "alphabet": raw["alphabet"]}


# ================================================================================================ source
class AudioSourceConfig(StrictConfig):
    path: str = Field(DEFAULT_NPZ, description="SYNTHETIC tone-sequence fixture (.npz written by examples/make_domain_fixtures.py)")
    n: int | None = Field(None, ge=1, title="use the first n clips (empty = all)")
    n_examples: int = Field(6, ge=1, le=12, title="clips recorded for the waveform inspector")


@register
class AudioSource(DomainOperation):
    type = "domain.audio_source"
    inputs = ()
    outputs = ("audio",)
    out_kinds = {"audio": AUDIO_BATCH}
    Config = AudioSourceConfig
    summary_kind = "audio_data"

    def infer(self, cfg, ins, node_id):
        if not cfg.path or not resolve_path(cfg.path).is_file():
            raise OpError("E_FIXTURE_MISSING", f"Fixture '{cfg.path}' was not found. Generate the SYNTHETIC fixtures with `python examples/make_domain_fixtures.py`.", None, [Fix("Run python examples/make_domain_fixtures.py")])
        return {"audio": VType(AUDIO_BATCH, _info(peek_spec(cfg.path)))}

    def execute(self, cfg, ins, ctx):
        p = fixture_path(cfg.path)
        clips, spec = A.load_npz(p, cfg.n)
        sha = sha256_file(p)
        data = _batch_data(clips, spec["alphabet"], {"source": {"path": str(p), "sha256": sha, "synthetic": True}})
        t = A.teaching_signal()
        assert t.samples_per_channel == A.TEACHING_SAMPLES_PER_CHANNEL
        first = clips[0]
        summary = {"path": str(p), "sha256": sha, "synthetic": True, "note": SYNTH, "contract": data["contract"], "clips": [clip_record(c) for c in clips[:cfg.n_examples]],
                   "lengthTable": [{"id": c.id, "samples": c.samples_per_channel, "seconds": c.duration} for c in clips[:20]],
                   "teaching": {**A.contract(t), "what": "a 2 s, 440 Hz sine generated in code at 16,000 Hz: exactly 32,000 samples per channel", "envelope": envelope(t.wave[0].numpy()),
                                "rawHead": [round(float(v), 4) for v in t.wave[0, :480]], "expectedSamples": A.TEACHING_SAMPLES_PER_CHANNEL},
                   "firstClipIsTeachingLength": first.samples_per_channel == A.TEACHING_SAMPLES_PER_CHANNEL, "timeAxis": "t = sample_index / sample_rate",
                   "provenance": {"source": "decoded from the fixture file (int16 PCM / 32767)"}}
        return {"audio": value(AUDIO_BATCH, data, {"clips": clips, "alphabet": spec["alphabet"]})}, summary

    def explain(self, cfg, inputs, outputs):
        return {"equation": "samples_per_channel = round(duration_seconds x sample_rate);  t = i / sample_rate",
                "rule": "The contract carries sample rate, channel count and length. A 2 s clip at 16,000 Hz has exactly 32,000 samples per channel.", "note": SYNTH}


# ================================================================================================ resample
class ResampleConfig(StrictConfig):
    target_rate: int = Field(16000, ge=1000, le=96000, title="target sample rate (Hz)")


@register
class AudioResample(DomainOperation):
    type = "domain.audio_resample"
    inputs = ("audio",)
    outputs = ("audio",)
    in_kinds = {"audio": AUDIO_BATCH}
    out_kinds = {"audio": AUDIO_BATCH}
    Config = ResampleConfig
    summary_kind = "audio_data"

    def infer(self, cfg, ins, node_id):
        i = dict(ins["audio"].info)
        old = i["sampleRate"]
        if old:
            for k in ("minSamples", "maxSamples"):
                if i.get(k):
                    i[k] = math.ceil(i[k] * cfg.target_rate / old)
        i["sampleRate"] = cfg.target_rate
        return {"audio": VType(AUDIO_BATCH, i)}

    def execute(self, cfg, ins, ctx):
        d = ins["audio"]
        clips = [A.resample(c, cfg.target_rate) for c in d.obj["clips"]]
        data = _batch_data(clips, d.obj["alphabet"], {"source": d.data.get("source"), "resampledFrom": d.data["contract"]["sampleRate"]})
        summary = {"contract": data["contract"], "from": d.data["contract"]["sampleRate"], "to": cfg.target_rate, "clips": [clip_record(c) for c in clips[:4]],
                   "lengthTable": [{"id": c.id, "samples": c.samples_per_channel, "seconds": c.duration} for c in clips[:20]],
                   "provenance": {"source": f"torchaudio.functional.resample {d.data['contract']['sampleRate']} -> {cfg.target_rate} Hz; lengths are ceil(N x target / source)"}}
        return {"audio": value(AUDIO_BATCH, data, {"clips": clips, "alphabet": d.obj["alphabet"]})}, summary

    def explain(self, cfg, inputs, outputs):
        return {"equation": "N' = ceil(N x target_rate / source_rate)", "rule": "Band-limited resampling (torchaudio.functional.resample). Duration is preserved; the sample count changes."}


# ================================================================================================ features
class AudioFeaturesConfig(FE.FeatureConfig):
    expected_sample_rate: int = Field(16000, ge=1000, le=96000, title="sample rate these features are designed for (must equal the wire's)")
    n_examples: int = Field(4, ge=1, le=10, title="clips recorded with their spectrogram")


@register
class AudioFeatures(DomainOperation):
    type = "domain.audio_features"
    inputs = ("audio",)
    outputs = ("features",)
    in_kinds = {"audio": AUDIO_BATCH}
    out_kinds = {"features": FEATURE_BATCH}
    Config = AudioFeaturesConfig
    summary_kind = "audio_features"

    def infer(self, cfg, ins, node_id):
        i = ins["audio"].info
        sr = i["sampleRate"]
        if sr is not None and sr != cfg.expected_sample_rate:
            raise OpError("E_AUDIO_SAMPLE_RATE", f"The audio wire is {sr} Hz but these features are configured for {cfg.expected_sample_rate} Hz (window and hop are specified in samples, so every time scale would be wrong).", "audio",
                          [Fix(f"Insert a 'Resample' node with target_rate={cfg.expected_sample_rate} before this node"), Fix(f"Or set expected_sample_rate to {sr}", None, "expected_sample_rate", sr)])
        for code, msg in FE.check_config(cfg, sr or cfg.expected_sample_rate):
            raise OpError(code, msg, None)
        mn = i.get("minSamples")
        if not cfg.center and mn is not None and mn < cfg.n_fft:
            raise OpError("E_AUDIO_TOO_SHORT", f"With center=false a clip needs at least n_fft={cfg.n_fft} samples to give one frame; the shortest clip has {mn}.", "audio")
        if cfg.center and cfg.pad_mode == "reflect" and mn is not None and mn <= cfg.n_fft // 2:
            raise OpError("E_AUDIO_TOO_SHORT", "Reflect padding requires more samples than n_fft//2.", "audio")
        frames = None if mn is None else (FE.frame_count(mn, cfg.n_fft, cfg.hop_length, cfg.center), FE.frame_count(i["maxSamples"], cfg.n_fft, cfg.hop_length, cfg.center))
        return {"features": VType(FEATURE_BATCH, {"sampleRate": sr, "nMels": cfg.n_mels, "hop": cfg.hop_length, "framesMin": frames and frames[0], "framesMax": frames and frames[1], "alphabet": i.get("alphabet")})}

    def execute(self, cfg, ins, ctx):
        d = ins["audio"]
        clips: list[A.Clip] = d.obj["clips"]
        sr = clips[0].sample_rate
        if sr != cfg.expected_sample_rate:
            raise ExecutionError("E_AUDIO_SAMPLE_RATE", f"audio is {sr} Hz, features are configured for {cfg.expected_sample_rate} Hz")
        feats, table = [], []
        for c in clips:
            f = FE.features_for(c.wave, cfg, sr)
            want = FE.frame_count(c.samples_per_channel, cfg.n_fft, cfg.hop_length, cfg.center)
            if f.shape[0] != want:
                raise ExecutionError("E_AUDIO_FRAME_FORMULA", f"{c.id}: the frame-count formula gives {want} frames but the transform produced {f.shape[0]}")
            feats.append(f)
            table.append({"id": c.id, "samples": c.samples_per_channel, "formulaFrames": want, "actualFrames": int(f.shape[0])})
        x, lengths, mask = FE.pad_features(feats)
        if not bool((mask.sum(1) == lengths).all()):
            raise ExecutionError("E_AUDIO_MASK", "mask does not cover exactly the real frames")
        w = clips[0].wave.mean(0)
        stft = torch.stft(w, n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length, window=FE.window_fn(cfg.window)(cfg.win_length), center=cfg.center,
                          pad_mode=cfg.pad_mode, return_complex=True)
        hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1)
        mel = lambda f: 2595.0 * math.log10(1 + f / 700.0)
        fmax = cfg.f_max or sr / 2
        pts = np.linspace(mel(cfg.f_min), mel(fmax), cfg.n_mels + 2)
        centres = [round(hz(m), 1) for m in pts[1:-1]]
        data = {"contract": {"n": len(clips), "sampleRate": sr, "nMels": cfg.n_mels, "hop": cfg.hop_length, "framesMin": int(lengths.min()), "framesMax": int(lengths.max()), "paddedFrames": int(x.shape[1]),
                             "alphabet": d.obj["alphabet"]}, "source": d.data.get("source"), "config": FE.describe(cfg, sr)}
        ex = []
        for i in range(min(cfg.n_examples, len(clips))):
            c = clips[i]
            ex.append({"id": c.id, "text": c.text, "samples": c.samples_per_channel, "frames": int(feats[i].shape[0]), "spectrogram": [[round(float(v), 2) for v in row] for row in feats[i].tolist()],
                       "frameTimes": [round(FE.frame_time(t, cfg.hop_length, sr, cfg.center, cfg.n_fft)[1], 4) for t in range(feats[i].shape[0])], "envelope": envelope(c.wave[0].numpy(), 400)})
        summary = {"config": FE.describe(cfg, sr), "contract": data["contract"], "frameTable": table[:30], "formulaChecks": {"clips": len(table), "allEqual": all(t["formulaFrames"] == t["actualFrames"] for t in table)},
                   "stftCheck": {"clip": clips[0].id, "torchStftFrames": int(stft.shape[-1]), "formulaFrames": table[0]["formulaFrames"], "freqBins": int(stft.shape[-2])},
                   "batch": {"shape": list(x.shape), "lengths": lengths.tolist()[:40], "maskTrueCounts": mask.sum(1).tolist()[:40], "paddingFraction": float(1 - mask.float().mean()),
                             "note": "features are computed per clip BEFORE padding, so padding never leaks into a clip's frames; the mask is True on real frames"},
                   "melCentresHz": centres, "examples": ex, "durationNote": "frame count is computed from the declared window/hop/padding, not from duration alone",
                   "provenance": {"source": d.data.get("source"), "library": f"torchaudio {__import__('torchaudio').__version__} MelSpectrogram (torch.stft)", "floor": cfg.floor if cfg.log else None}}
        return {"features": value(FEATURE_BATCH, data, {"feats": feats, "texts": [c.text for c in clips], "alphabet": d.obj["alphabet"], "cfg": cfg, "sr": sr, "clips": clips})}, summary

    def explain(self, cfg, inputs, outputs):
        return {"equation": FE.describe(cfg, cfg.expected_sample_rate)["formula"], "rule": f"window {cfg.win_length} samples ({cfg.window}), hop {cfg.hop_length}, n_fft {cfg.n_fft}, {cfg.n_mels} mel bands, {'log' if cfg.log else 'linear'} {'power' if cfg.power == 2 else 'magnitude'}.",
                "note": "The frame count follows this configuration and each clip's sample count; it is verified against the real STFT at run time."}


# ================================================================================================ CTC
class CTCConfig(StrictConfig):
    resume_model_id: str | None = Field(None, pattern=r"^[0-9a-f]{64}$", description="Internal model manifest identity to resume at a completed epoch; epochs is the total target.")
    epochs: int = Field(60, ge=1, le=400)
    lr: float = Field(5e-3, gt=0)
    hidden: int = Field(48, ge=4, le=256)
    batch_size: int = Field(8, ge=1, le=256)
    val_fraction: float = Field(0.25, gt=0, lt=1)
    seed: int = 0
    reduction: Literal["mean", "sum"] = Field("mean", title="CTCLoss reduction ('mean' divides each loss by its target length, then averages)")
    zero_infinity: bool = Field(False, title="zero_infinity (zero out infinite losses from infeasible alignments)")
    n_inspect: int = Field(6, ge=1, le=12)


@register
class SpeechCTC(DomainOperation):
    type = "domain.speech_ctc"
    inputs = ("features",)
    outputs = ("report",)
    in_kinds = {"features": FEATURE_BATCH}
    out_kinds = {"report": SPEECH_REPORT}
    Config = CTCConfig
    summary_kind = "speech_ctc"

    def infer(self, cfg, ins, node_id):
        if not ins["features"].info.get("alphabet"):
            raise OpError("E_CTC_NO_ALPHABET", "The feature wire carries no label alphabet, so there is nothing to recognise.", "features")
        return {"report": VType(SPEECH_REPORT, {})}

    def execute(self, cfg, ins, ctx):
        d = ins["features"]
        o = d.obj
        signature = CP.fingerprint(cfg.model_dump(exclude={"epochs", "n_inspect", "resume_model_id"}),
                                   {"data": d.data, "texts": o["texts"], "alphabet": o["alphabet"]}, o["feats"])
        saved = CP.resume(ctx, cfg.resume_model_id, "speech", signature)
        try:
            res = CT.train_and_eval(o["feats"], o["texts"], o["alphabet"], epochs=cfg.epochs, lr=cfg.lr, hidden=cfg.hidden, batch_size=cfg.batch_size, seed=cfg.seed, val_fraction=cfg.val_fraction,
                                    n_inspect=cfg.n_inspect, hop_seconds=o["cfg"].hop_length / o["sr"], reduction=cfg.reduction, zero_infinity=cfg.zero_infinity, resume_state=saved)
        except ValueError as e:
            if str(e).startswith("E_CTC_ALIGNMENT_INFEASIBLE:"):
                raise ExecutionError("E_CTC_ALIGNMENT_INFEASIBLE", str(e).split(":", 1)[1]) from e
            raise
        fc = o["cfg"]
        sr = o["sr"]
        recs = []
        for x in res["records"]:
            c = o["clips"][x["index"]]
            f = x["tokenFrames"]
            # an output frame f is produced by the convolution centred on input frame STRIDE*f
            to_t = lambda k: FE.frame_time(CT.STRIDE * k, fc.hop_length, sr, fc.center, fc.n_fft)[1]
            recs.append({**{k: v for k, v in x.items() if k != "spectrogram"}, "id": c.id, "samples": c.samples_per_channel, "spectrogram": [[round(float(v), 2) for v in row] for row in x["spectrogram"].tolist()],
                         "envelope": envelope(c.wave[0].numpy(), 400), "tokenTimes": [round(to_t(k), 4) for k in f], "frameTimes": [round(to_t(k), 4) for k in range(x["outFrames"])],
                         "gold": c.segments and [{"symbol": s["symbol"], "t0": s["start"] / sr, "t1": s["end"] / sr} for s in c.segments], "durationSeconds": c.duration})
        rates = res["rates"]
        summary = {"config": cfg.model_dump(), "nParams": res["nParams"], "curve": res["curve"], "alphabet": res["alphabet"], "vocabulary": {"blank": 0, **{c: i + 1 for i, c in enumerate(res["alphabet"])}},
                   "ctc": {**res["ctc"], "inputLengths": "frames after the 2x time downsampling: floor((T + 2*1 - 3) / 2) + 1", "targetLengths": "label count per transcript (the separator tone is a label)",
                           "constraint": "an alignment needs at least (labels + adjacent repeats) frames", "perClip": [{"clip": x["id"], "inputFrames": x["frames"], "outputFrames": x["outFrames"], "targetLength": x["targetLength"], "minFramesNeeded": x["minFramesNeeded"]} for x in recs]},
                   "normalization": res["normalization"], "rates": {**rates, "provenance": "corpus-level: total edits / total reference length over the validation clips; CER counts the separator as a character, WER splits on it"},
                   "samples": recs, "split": {"seed": cfg.seed, "nVal": len(res["valIdx"]), "nTrain": len(res["trainIdx"]), "valIndices": res["valIdx"]}, "seconds": res["seconds"], "synthetic": True, "note": SYNTH,
                   "contract": d.data["contract"], "provenance": {"source": d.data.get("source"), "torch": torch.__version__, "model": "Conv1d(stride 2) -> BiGRU (packed) -> Linear; torch.nn.CTCLoss", "decoding": "greedy best path (argmax, collapse repeats, drop blanks)"}}
        summary["checkpoint"] = CP.persist(ctx, "speech", signature, res["trainingState"],
            {"architecture": {"n_mels": fc.n_mels, "vocab_with_blank": len(o["alphabet"]) + 1, "hidden": cfg.hidden},
             "sampleRate": sr, "features": {k: v for k, v in fc.model_dump().items() if k in FE.FeatureConfig.model_fields},
             "alphabet": o["alphabet"], "channels": d.data["contract"].get("channels", 1), "channelPolicy": "mean to mono", "blank": 0,
             "normalization": "training-only per-mel mean/std in checkpoint", "decoding": "greedy; onsets are not forced alignment"},
            d.data.get("source"), cfg.resume_model_id, {"samples": o["clips"][res["valIdx"][0]].wave.tolist(), "sampleRate": sr})
        return {"report": value(SPEECH_REPORT, {"metrics": {"cer": rates["cer"]["rate"], "wer": rates["wer"]["rate"]}, "nParams": res["nParams"]})}, summary

    def explain(self, cfg, inputs, outputs):
        return {"equation": "loss = CTCLoss(log_softmax(logits), targets, input_lengths, target_lengths; blank=0, reduction=" + cfg.reduction + ")",
                "rule": "CTC needs no frame-level labels: it sums over every alignment of the labels to the frames, with blanks between repeated labels. Greedy decoding takes the best frame path, collapses repeats and drops blanks.",
                "note": "Error alignment is a unit-cost edit-distance backtrace: substitutions, deletions and insertions are shown per utterance."}
