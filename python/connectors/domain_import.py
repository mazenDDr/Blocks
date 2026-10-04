"""Bounded local domain dataset conversion. Originals are read once and hashed; no downloads or installation."""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from tabular.core import ExecutionError, resolve_path

MAX_INPUT = 64 * 1024 * 1024
MAX_JSON = 8 * 1024 * 1024
MAX_DECODED = 64 * 1024 * 1024
MAX_ITEMS = 256


def fail(message, code="E_DATASET_FORMAT"):
    raise ExecutionError(code, message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dumps(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class Reader:
    def __init__(self):
        self.files = []
        self.blobs = []
        self.total = 0

    def read(self, path, cap=MAX_INPUT):
        path = resolve_path(str(path))
        if not path.is_file():
            fail(f"Source file not found: {path}", "E_DATASET_SOURCE")
        with path.open("rb") as f:
            raw = f.read(min(cap, MAX_INPUT - self.total) + 1)
        if len(raw) > cap or self.total + len(raw) > MAX_INPUT:
            fail("Source bytes exceed the declared import limit.", "E_DATASET_BOUNDS")
        self.total += len(raw)
        self.files.append({"path": str(path), "sha256": sha(raw), "bytes": len(raw)})
        self.blobs.append(raw)
        return raw


def child(root, name):
    p = Path(name)
    if p.is_absolute() or ".." in p.parts:
        fail("Dataset member paths must be relative and contained in the source directory.", "E_DATASET_PATH")
    root = resolve_path(str(root))
    dest = (root / p).resolve()
    if not dest.is_relative_to(root):
        fail("Dataset member resolves outside its source directory.", "E_DATASET_PATH")
    return dest


def bounded(n):
    if not 2 <= n <= MAX_ITEMS:
        fail(f"Import 2–{MAX_ITEMS} records; larger datasets need a separate streaming importer.", "E_DATASET_BOUNDS")


def check_compressed_rle(text, pixels):
    """Validate COCO's signed/delta varints before handing counts to the native decoder."""
    counts, pos = [], 0
    while pos < len(text):
        value, shift = 0, 0
        while True:
            if pos >= len(text) or shift > 60:
                fail("Malformed compressed RLE.", "E_DATASET_MASK")
            c = ord(text[pos])-48; pos += 1
            if not 0 <= c <= 63:
                fail("Malformed compressed RLE character.", "E_DATASET_MASK")
            value |= (c & 31) << shift; shift += 5
            if not c & 32:
                if c & 16:
                    value |= -1 << shift
                break
        if len(counts) > 2:
            value += counts[-2]
        if value < 0 or value > pixels:
            fail("Invalid compressed RLE run length.", "E_DATASET_MASK")
        counts.append(value)
        if len(counts) > pixels+1:
            fail("Too many RLE runs.", "E_DATASET_MASK")
    if sum(counts) != pixels:
        fail("Compressed RLE must cover exactly H×W pixels.", "E_DATASET_MASK")


def npz(arrays):
    if sum(a.nbytes for a in arrays.values()) > MAX_DECODED:
        fail("Decoded canonical arrays exceed 64 MiB.", "E_DATASET_BOUNDS")
    b = io.BytesIO()
    np.savez_compressed(b, **arrays)
    return b.getvalue()


def coco(req, reader):
    from pycocotools import mask as M

    raw = json.loads(reader.read(req["path"], MAX_JSON))
    images, cats, anns = raw["images"], raw["categories"], raw["annotations"]
    bounded(len(images))
    if not 1 <= len(cats) <= 4 or len({c["id"] for c in cats}) != len(cats) or len({i["id"] for i in images}) != len(images):
        fail("Category and image IDs must be unique; this bounded segmenter supports 1–4 categories.")
    cats = sorted(cats, key=lambda c: c["id"])
    classes = [c["name"] for c in cats]
    if any(not isinstance(n, str) or not n for n in classes) or len(set(classes)) != len(classes):
        fail("Category names must be unique nonempty strings.")
    mapping = {c["id"]: i for i, c in enumerate(cats)}
    names = cats[0].get("keypoints", [])
    if not isinstance(names, list) or len(names) > 128 or len(set(names)) != len(names):
        fail("Require up to 128 unique keypoint names.")
    if any(c.get("keypoints", []) != names for c in cats):
        fail("This contract requires one common keypoint schema across categories.")
    pairs = req.get("flip_pairs", [])
    if any(len(p) != 2 or min(p) < 0 or max(p) >= len(names) for p in pairs) or len({x for p in pairs for x in p}) != 2 * len(pairs):
        fail("Flip pairs must be disjoint indices into the declared keypoint schema.")
    by_image = {i["id"]: [] for i in images}
    if len({a["id"] for a in anns}) != len(anns):
        fail("Annotation IDs must be unique.")
    for a in anns:
        if a["image_id"] not in by_image or a["category_id"] not in mapping:
            fail("Annotation references an unknown image or category.")
        if a.get("iscrowd", 0):
            fail("Crowd regions are not supported by this training/evaluation contract.", "E_DATASET_UNSUPPORTED")
        by_image[a["image_id"]].append(a)
    h, w = int(images[0]["height"]), int(images[0]["width"])
    counts = [len(by_image[i["id"]]) for i in images]
    slots = max(1, max(counts))
    if not 4 <= h <= 512 or not 4 <= w <= 512 or slots > 64 or any(int(i["height"]) != h or int(i["width"]) != w for i in images):
        fail("Images must share H×W (4–512 each); at most 64 instances/image. No implicit geometry.", "E_DATASET_BOUNDS")
    if len(images) * (h*w*(3+slots) + slots*(24+9*len(names))) > MAX_DECODED:
        fail("Decoded images/masks exceed 64 MiB.", "E_DATASET_BOUNDS")
    boxes = np.zeros((len(images), slots, 4), np.float32)
    labels = np.full((len(images), slots), -1, np.int64)
    masks = np.zeros((len(images), slots, h, w), bool)
    kps = np.zeros((len(images), slots, len(names), 2), np.float32)
    vis = np.zeros(kps.shape[:-1], np.int8)
    pixels = []
    metadata = []
    for i, row in enumerate(images):
        encoded = reader.read(child(req["root"], row["file_name"]))
        with Image.open(io.BytesIO(encoded)) as im:
            if im.format not in ("PNG", "JPEG") or im.size != (w, h) or im.mode != "RGB":
                fail("COCO image must be RGB PNG/JPEG at its declared dimensions.")
            pixels.append(np.asarray(im).copy())
        metadata.append({"imageId": row["id"], "fileName": row["file_name"], "cocoLicenseId": row.get("license"), "annotationIds": []})
        for j, a in enumerate(by_image[row["id"]]):
            b = np.asarray(a["bbox"], dtype=np.float32)
            if b.shape != (4,) or not np.isfinite(b).all() or min(b[2:]) <= 0 or min(b[:2]) < 0 or b[0]+b[2] > w or b[1]+b[3] > h:
                fail("COCO bbox must be finite, nondegenerate xywh within its image.", "E_DATASET_BOX")
            seg = a.get("segmentation")
            if isinstance(seg, list):
                if not seg or any(len(p) < 6 or len(p) % 2 or not np.isfinite(np.asarray(p, float)).all() for p in seg):
                    fail("Require nonempty finite COCO polygons.", "E_DATASET_MASK")
                if any(min(p) < 0 or max(p[::2]) > w or max(p[1::2]) > h for p in seg):
                    fail("Polygon coordinates must lie within the declared image.", "E_DATASET_MASK")
                rle = M.merge(M.frPyObjects(seg, h, w))
            elif isinstance(seg, dict):
                if seg.get("size") != [h, w]:
                    fail("RLE dimensions differ from the image.", "E_DATASET_MASK")
                if isinstance(seg.get("counts"), list):
                    counts_ = seg["counts"]
                    if any(type(n) is not int or n < 0 for n in counts_) or len(counts_) > h*w+1 or sum(counts_) != h*w:
                        fail("Uncompressed RLE must cover exactly H×W pixels.", "E_DATASET_MASK")
                    rle = M.frPyObjects(seg, h, w)
                elif isinstance(seg.get("counts"), str):
                    check_compressed_rle(seg["counts"], h*w)
                    rle = {**seg, "counts": seg["counts"].encode("ascii")}
                else:
                    fail("Invalid COCO RLE counts.", "E_DATASET_MASK")
            else:
                fail("Every annotation needs a segmentation mask; box-only COCO is not implemented.", "E_DATASET_UNSUPPORTED")
            decoded = M.decode(rle)
            if decoded.shape != (h, w) or not decoded.any():
                fail("Mask must be nonempty H×W.", "E_DATASET_MASK")
            boxes[i, j] = [b[0], b[1], b[0]+b[2], b[1]+b[3]]
            labels[i, j] = mapping[a["category_id"]]
            masks[i, j] = decoded.astype(bool)
            if names:
                k = np.asarray(a.get("keypoints", [0] * (3*len(names))), float)
                if k.shape != (3*len(names),) or not np.isfinite(k).all():
                    fail("Keypoints must match the common schema.", "E_DATASET_KEYPOINT")
                k = k.reshape(-1, 3)
                if not np.isin(k[:, 2], [0, 1, 2]).all() or np.any((k[:, 2] > 0) & ((k[:, 0] < 0) | (k[:, 0] >= w) | (k[:, 1] < 0) | (k[:, 1] >= h))):
                    fail("Visible/occluded keypoints must lie within the image, with visibility 0/1/2.", "E_DATASET_KEYPOINT")
                kps[i, j], vis[i, j] = k[:, :2], k[:, 2]
            metadata[i]["annotationIds"].append(a["id"])
    spec = {"n": len(images), "size": [h, w], "classes": classes, "keypointNames": names, "flipPairs": pairs,
            "ids": [str(i["id"]) for i in images], "synthetic": req["synthetic"], "categoryIds": [c["id"] for c in cats],
            "boxPolicy": "COCO xywh converted exactly to xyxy; no clipping or mask refit", "overlapPolicy": "separate instance masks retained; semantic target uses later annotation last", "sampleMetadata": metadata, "cocoLicenses": raw.get("licenses", [])}
    return npz({"images": np.stack(pixels), "boxes": boxes, "labels": labels, "masks": masks, "keypoints": kps, "visibility": vis, "spec": np.array(dumps(spec))}), ".npz", spec


def conll(req, reader):
    from nlp.iob import validate_iob2

    content = reader.read(req["path"], MAX_JSON).decode("utf-8")
    records, tokens, tags, lines = [], [], [], []
    def flush():
        if not tokens:
            return
        bad = validate_iob2(tags)
        if bad:
            fail(f"Sentence {len(records)} has invalid IOB2: {bad[0]['message']}", "E_DATASET_IOB2")
        text = " ".join(tokens)
        if len(text) > 4000:
            fail("Sentence exceeds 4000 characters.", "E_DATASET_BOUNDS")
        spans, offset, current = [], 0, None
        for token, tag in zip(tokens, tags):
            if tag == "O" or tag.startswith("B-"):
                if current:
                    spans.append(current)
                current = None
            if tag.startswith("B-"):
                current = {"start": offset, "end": offset+len(token), "label": tag[2:]}
            elif tag.startswith("I-"):
                current["end"] = offset+len(token)
            offset += len(token)+1
        if current:
            spans.append(current)
        records.append({"id": f"conll_{len(records):04d}", "text": text, "spans": spans, "synthetic": req["synthetic"], "sourceLines": list(lines), "originalTokens": list(tokens), "originalTags": list(tags)})
        tokens.clear(); tags.clear(); lines.clear()
        if len(records) > MAX_ITEMS:
            fail("Too many sentences.", "E_DATASET_BOUNDS")
    for n, line in enumerate(content.splitlines(), 1):
        if not line.strip() or line.startswith("-DOCSTART-"):
            flush(); continue
        fields = line.split()
        t, tag = fields[req.get("token_column", 0)], fields[req.get("label_column", -1)]
        if not t:
            fail("Empty CoNLL token.")
        tokens.append(t); tags.append(tag); lines.append(n)
    flush(); bounded(len(records))
    if len({s["label"] for r in records for s in r["spans"]}) > 32:
        fail("Bounded token classification supports at most 32 entity types.", "E_DATASET_BOUNDS")
    return ("".join(dumps(r)+"\n" for r in records).encode(), ".jsonl", {"n": len(records), "types": sorted({s["label"] for r in records for s in r["spans"]}),
            "textPolicy": "UTF-8, blank-line sentences; tokens joined by one ASCII space; character offsets refer to reconstructed text, not original file bytes", "scheme": "IOB2 (strict); DOCSTART is a boundary"})


def wav(req, reader):
    from scipy.io import wavfile

    entries = json.loads(reader.read(req["path"], MAX_JSON))
    if not isinstance(entries, list):
        fail("WAV manifest must be a JSON array of {id, file, text}.")
    bounded(len(entries))
    if len({e["id"] for e in entries}) != len(entries):
        fail("Clip IDs must be unique.")
    waves, rates, channels, lengths, texts = [], [], [], [], []
    for e in entries:
        if not isinstance(e["text"], str) or not e["text"].strip() or len(e["text"]) > 256:
            fail("Transcripts must be nonempty strings of at most 256 characters.")
        sr, x = wavfile.read(io.BytesIO(reader.read(child(req["root"], e["file"]))))
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or not 1 <= x.shape[1] <= 4 or not 512 <= len(x) <= 32000 or not 1000 <= sr <= 96000:
            fail("Require 512–32,000 samples/channel, 1–4 channels and 1–96 kHz; no implicit trim/resample.", "E_DATASET_BOUNDS")
        dtype = x.dtype
        if dtype == np.uint8:
            x = (x.astype(np.float32)-128)/128
        elif np.issubdtype(dtype, np.signedinteger):
            x = x.astype(np.float32) / float(2 ** (dtype.itemsize*8-1))
        elif np.issubdtype(dtype, np.floating):
            x = x.astype(np.float32)
        else:
            fail("Unsupported WAV sample encoding.", "E_DATASET_UNSUPPORTED")
        if not np.isfinite(x).all() or np.abs(x).max() > 1:
            fail("PCM must be finite and normalized in [-1,1]; out-of-range float WAV is refused.", "E_DATASET_AUDIO")
        waves.append(x.T.copy()); rates.append(int(sr)); channels.append(x.shape[1]); lengths.append(x.shape[0]); texts.append(e["text"])
    if len(set(rates)) != 1:
        fail("Mixed sample rates need explicit preprocessing before import.", "E_DATASET_AUDIO_RATE")
    if len(set(channels)) != 1:
        fail("Mixed channel counts need explicit preprocessing before import.", "E_DATASET_AUDIO_CHANNELS")
    if len(set("".join(texts))) > 64:
        fail("Bounded CTC supports at most 64 distinct transcript characters.", "E_DATASET_BOUNDS")
    padded = np.zeros((len(entries), channels[0], max(lengths)), np.float32)
    for i, x in enumerate(waves):
        padded[i, :, :x.shape[1]] = x
    spec = {"n": len(entries), "synthetic": req["synthetic"], "sampleRate": rates[0], "channels": channels[0], "ids": [str(e["id"]) for e in entries],
            "alphabet": sorted(set("".join(texts))), "durationsSeconds": sorted({n/rates[0] for n in lengths}), "dtype": "normalized float32 PCM (N,C,L), preserved channels", "normalization": "unsigned8: (x-128)/128; signed integers: x/2^(bits-1), including left-justified 24-bit; float: unchanged", "segments": "not recorded; transcripts provide no timestamps"}
    return npz({"waveforms": padded, "lengths": np.array(lengths, np.int32), "transcripts": np.array(texts), "segments": np.array(dumps([None]*len(entries))), "spec": np.array(dumps(spec))}), ".npz", spec


def import_dataset(store, req):
    reader = Reader()
    try:
        payload, ext, contract = {"coco": coco, "conll": conll, "wav": wav}[req["kind"]](req, reader)
        if len(payload) > MAX_DECODED:
            fail("Canonical payload exceeds 64 MiB.", "E_DATASET_BOUNDS")
        m = {"format": "void-domain-dataset/1", "kind": req["kind"], "family": {"coco":"vision", "conll":"nlp", "wav":"speech"}[req["kind"]],
             "synthetic": req["synthetic"], "license": {"declaration": req["license"], "authority": "user supplied; not independently verified"},
             "request": req, "sources": reader.files, "contract": contract, "payloadSha256": sha(payload), "extension": ext,
             "converterSha256": sha(Path(__file__).read_bytes())}
        raw = dumps(m).encode(); identity = sha(raw)
        folder = store.root / "domain-datasets" / identity
        folder.mkdir(parents=True, exist_ok=True)
        for b in [*reader.blobs, payload, raw]:
            store.put_bytes(b)
        # Atomic payload then manifest: listing never treats an incomplete import as ready.
        for name, b in [("data"+ext, payload), ("manifest.json", raw)]:
            fd, tmp = tempfile.mkstemp(dir=folder)
            with os.fdopen(fd, "wb") as f:
                f.write(b)
            os.replace(tmp, folder/name)
        return {"id": identity, "path": str(folder/("data"+ext)), **m}
    except (KeyError, ValueError, TypeError, OSError, IndexError, OverflowError) as e:
        raise ExecutionError("E_DATASET_FORMAT", f"Cannot convert dataset: {type(e).__name__}: {str(e)[:250]}") from e


def provenance(path, synthetic, note):
    path = Path(path).resolve()
    payload_sha = sha(path.read_bytes())
    out = {"path": str(path), "sha256": payload_sha, "synthetic": synthetic, "license": None,
           "note": "Source declares SYNTHETIC data; file hash recorded." if synthetic is True else "Source declares synthetic=false." if synthetic is False else "Source synthetic status is not declared."}
    if path.parent.parent.name == "domain-datasets":
        manifest = path.parent / "manifest.json"
        if not manifest.is_file():
            fail("Imported dataset manifest is missing.", "E_DATASET_INTEGRITY")
        try:
            raw = manifest.read_bytes(); m = json.loads(raw)
            required = {"format", "payloadSha256", "extension", "synthetic", "license", "kind", "sources", "converterSha256"}
            if not isinstance(m, dict) or not required <= m.keys():
                fail("Imported manifest fields are missing.", "E_DATASET_INTEGRITY")
        except (ValueError, OSError) as e:
            raise ExecutionError("E_DATASET_INTEGRITY", "Cannot read imported dataset manifest.") from e
        if sha(raw) != path.parent.name or m["format"] != "void-domain-dataset/1" or m["payloadSha256"] != payload_sha or path.name != "data"+m["extension"]:
            fail("Imported dataset manifest or canonical payload hash differs.", "E_DATASET_INTEGRITY")
        out.update(synthetic=m["synthetic"], license=m["license"], datasetId=path.parent.name, importKind=m["kind"], sourceFiles=m["sources"],
                   converterSha256=m["converterSha256"], note=f"Imported {m['kind']} dataset; synthetic={str(m['synthetic']).lower()} is user-declared; license is user-declared.")
    return out
