"""Bounded local inference from hash-verified internal native state dictionaries."""
from __future__ import annotations
import base64
import io
import math
import numpy as np
import torch
from PIL import Image
from tabular.core import ExecutionError, dumps
from . import checkpoints as CP
from .core import png_b64


def fail(message, code="E_DOMAIN_INPUT"):
    raise ExecutionError(code, message)


def make_model(m):
    arch = m["inference"]["architecture"]
    if m["family"] == "vision":
        from vision.segment import TinyFCN
        return TinyFCN(**arch)
    if m["family"] == "nlp":
        from nlp.model import Tagger
        return Tagger(**arch)
    from speech.ctc import CTCModel
    return CTCModel(**arch)


def predict(store, identity, records):
    if not isinstance(records, list) or not 1 <= len(records) <= 4 or len(dumps(records).encode()) > 1_500_000:
        fail("Use 1–4 records within 1.5 MB.", "E_DOMAIN_INPUT_BOUNDS")
    m = CP.read_manifest(store, identity)
    state = CP.read_state(store, m)
    model = make_model(m)
    model.load_state_dict(state["weights"], strict=True); model.eval()
    cfg = m["inference"]
    out = []
    with torch.inference_mode():
        for record in records:
            if not isinstance(record, dict):
                fail("Each input must be a typed object.")
            if m["family"] == "vision":
                if set(record) != {"imagePng"} or not isinstance(record["imagePng"], str):
                    fail("Expected only imagePng: base64 RGB PNG at the pinned post-geometry size.")
                try:
                    raw = base64.b64decode(record["imagePng"], validate=True)
                    with Image.open(io.BytesIO(raw)) as im:
                        if im.format != "PNG" or im.mode != "RGB" or list(reversed(im.size)) != cfg["size"] or max(im.size) > 512:
                            fail("PNG must be RGB at the pinned H×W (at most 512 each); geometry is explicit.")
                        x = torch.from_numpy(np.array(im)).permute(2, 0, 1)
                except ExecutionError:
                    raise
                except Exception as e:
                    raise ExecutionError("E_DOMAIN_INPUT", "Invalid RGB PNG.") from e
                from vision.segment import prep, components, PALETTE
                logits = model(prep(x[None]))[0]; mask = logits.argmax(0); prob = logits.softmax(0)
                boxes, labels, scores = components(prob, mask)
                out.append({"maskPng": png_b64(mask.numpy(), PALETTE), "classes": cfg["classes"],
                            "boxes": boxes.tolist(), "labels": labels.tolist(), "scores": scores.tolist(),
                            "detectionMeaning": "derived connected components of segmentation", "size": cfg["size"]})
            elif m["family"] == "nlp":
                if set(record) != {"text"} or not isinstance(record["text"], str) or not record["text"].strip() or len(record["text"]) > 4000:
                    fail("Expected only nonempty text, at most 4000 characters.")
                from tokenizers import Tokenizer
                from nlp.subword import encode_example, batch_tensors, first_subword_positions
                from nlp.iob import validate_iob2
                tok = Tokenizer.from_str(cfg["tokenizerJson"])
                e = encode_example(tok, {"id": "request", "text": record["text"], "spans": []},
                                   {t:i for i,t in enumerate(cfg["labels"])}, cfg["maxLength"], cfg["labelPolicy"], cfg["ignoreIndex"])
                ids, attention, _ = batch_tensors([e], cfg["architecture"]["pad_id"], cfg["ignoreIndex"])
                lg = model(ids, attention)[0]; tags = [cfg["labels"][i] for i in lg.argmax(-1).tolist()]
                pos = first_subword_positions(e)
                words = [{"text": w["text"], "start": w["start"], "end": w["end"], "label": tags[pos[w["wordId"]]]} for w in e.words]
                out.append({"text": e.text, "tokens": e.tokens, "offsets": e.offsets, "attention": attention[0].tolist(),
                            "predictions": tags, "probabilities": lg.softmax(-1).tolist(), "words": words,
                            "invalidIob2": validate_iob2([w["label"] for w in words]), "truncatedWords": e.truncated_words,
                            "readout": cfg["readout"], "labelPolicy": cfg["labelPolicy"]})
            else:
                if set(record) != {"samples", "sampleRate"} or type(record["sampleRate"]) is not int or record["sampleRate"] != cfg["sampleRate"]:
                    fail("Expected samples [channel][sample] and exactly the pinned sampleRate.", "E_AUDIO_SAMPLE_RATE")
                channels = record["samples"]
                if not isinstance(channels, list) or not 1 <= len(channels) <= 4 or any(not isinstance(ch, list) for ch in channels):
                    fail("Audio requires 1–4 channels.")
                lengths = {len(ch) for ch in channels}
                if len(lengths) != 1 or not 1 <= next(iter(lengths)) <= 32000:
                    fail("Channels must have equal length, 1–32000 samples per channel.", "E_DOMAIN_INPUT_BOUNDS")
                if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1 for ch in channels for v in ch):
                    fail("Audio samples must be finite normalized PCM in [-1,1].")
                from speech.features import FeatureConfig, features_for, pad_features, frame_count, frame_time
                from speech.ctc import greedy_decode, STRIDE
                fc = FeatureConfig.model_validate(cfg["features"])
                n = len(channels[0]); frames = frame_count(n, fc.n_fft, fc.hop_length, fc.center)
                if frames < 1 or frames > 4096 or (fc.center and fc.pad_mode == "reflect" and n <= fc.n_fft // 2):
                    fail("Clip is too short for the pinned window or exceeds 4096 feature frames.", "E_DOMAIN_INPUT_BOUNDS")
                feat = features_for(torch.tensor(channels, dtype=torch.float32), fc, cfg["sampleRate"])
                x, lens, _ = pad_features([(feat - state["mean"]) / state["std"]])
                lp, ol = model(x, lens); tokens, at = greedy_decode(lp[0], int(ol[0]), cfg["blank"])
                out.append({"text": "".join(cfg["alphabet"][i-1] for i in tokens), "tokenIds": tokens, "tokenFrames": at,
                            "tokenTimes": [frame_time(STRIDE*t, fc.hop_length, cfg["sampleRate"], fc.center, fc.n_fft)[1] for t in at],
                            "featureFrames": int(lens[0]), "outputFrames": int(ol[0]), "decoding": cfg["decoding"],
                            "sampleRate": cfg["sampleRate"], "channels": len(channels), "channelPolicy": cfg["channelPolicy"]})
    return {"predictions": out, "provenance": {"modelId": identity, "checkpointSha256": m["checkpointSha256"],
            "runId": m["runId"], "graphHash": m["graphHash"], "node": m["node"], "epochs": m["epochs"], "source": m["source"],
            "family": m["family"], "location": "native CPU inference in local control process; no retraining"}}
