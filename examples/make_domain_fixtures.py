"""Deterministic, labelled-SYNTHETIC fixtures for the Milestone 6b domain workflows (vision, NLP, speech). Nothing here is measured data.

    python examples/make_domain_fixtures.py            # writes examples/fixtures/synthetic_ner.jsonl (committed) and examples/data/domain/*.npz (gitignored)

Every generator is a pure function of (n, seed): running it twice produces byte-identical files.

- vision: 48x48 RGB images with 1-3 drawn objects (class 0 = rectangle, class 1 = disk). Objects are drawn in order, so later ones occlude earlier ones;
  the instance id map, the boxes (tight bounds of the VISIBLE pixels, xyxy, pixel edges, x_max/y_max exclusive) and the keypoints are all derived from that
  final picture. Keypoints per object: 0 = left end, 1 = right end, 2 = centre (continuous pixel coordinates, pixel i spans [i, i+1)); visibility follows the
  COCO convention: 2 = labelled and visible, 1 = labelled but occluded by another object.
- NLP: template sentences with made-up names. Entity spans are CHARACTER offsets into the raw text (end exclusive); types PER, ORG, LOC.
- speech: 16 kHz mono tone sequences. Symbol -> tone: a 300 Hz, b 600 Hz, c 1000 Hz, d 1500 Hz, ' ' (word separator) 2200 Hz. Clip lengths 1.0, 1.5 or 2.0 s
  (16,000 / 24,000 / 32,000 samples); the true segment of every symbol (start/end sample) is stored next to the transcript."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "domain"
FIXTURES = HERE / "fixtures"
VISION_NPZ, AUDIO_NPZ, NER_JSONL = DATA / "synthetic_shapes_det.npz", DATA / "synthetic_tones.npz", FIXTURES / "synthetic_ner.jsonl"

# --------------------------------------------------------------------------------------------------------------------------------------- vision
VISION_CLASSES = ["rectangle", "disk"]
KEYPOINT_NAMES = ["left", "right", "center"]
FLIP_PAIRS = [[0, 1]]  # horizontal flip exchanges the left and right end points
MAX_OBJ = 3


def _draw(rng, size: int):
    yy, xx = np.mgrid[0:size, 0:size]
    base = 90 + 25 * (xx / size) * rng.uniform(-1, 1) + 25 * (yy / size) * rng.uniform(-1, 1)
    img = np.repeat(base[..., None], 3, axis=2) + rng.normal(0, 6, (size, size, 3))
    inst = np.zeros((size, size), np.uint8)
    objs = []
    n_obj = int(rng.integers(1, MAX_OBJ + 1))
    for k in range(n_obj):
        cls = int(rng.integers(0, 2))
        if cls == 0:
            w, h = int(rng.integers(9, 21)), int(rng.integers(9, 21))
            x0, y0 = int(rng.integers(1, size - w)), int(rng.integers(1, size - h))
            m = (xx >= x0) & (xx < x0 + w) & (yy >= y0) & (yy < y0 + h)
            cx, cy = x0 + w // 2, y0 + h // 2
            left, right = (x0 + 0.5, cy + 0.5), (x0 + w - 0.5, cy + 0.5)
            color = np.array([200, 70, 60]) + rng.integers(-25, 25, 3)
        else:
            r = int(rng.integers(5, 11))
            cx, cy = int(rng.integers(r + 1, size - r - 1)), int(rng.integers(r + 1, size - r - 1))
            m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            left, right = (cx - r + 0.5, cy + 0.5), (cx + r + 0.5, cy + 0.5)
            color = np.array([60, 190, 90]) + rng.integers(-25, 25, 3)
        inst[m] = k + 1
        img[m] = color + rng.normal(0, 4, (int(m.sum()), 3))
        objs.append((cls, [left, right, (cx + 0.5, cy + 0.5)]))
    return np.clip(img, 0, 255).astype(np.uint8), inst, objs


def vision_fixture(n: int = 96, size: int = 48, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    images = np.zeros((n, size, size, 3), np.uint8)
    inst = np.zeros((n, size, size), np.uint8)
    boxes = np.zeros((n, MAX_OBJ, 4), np.float32)
    labels = np.full((n, MAX_OBJ), -1, np.int16)
    kps = np.zeros((n, MAX_OBJ, 3, 2), np.float32)
    vis = np.zeros((n, MAX_OBJ, 3), np.int8)
    i = 0
    while i < n:
        img, im, objs = _draw(rng, size)
        keep = [k for k in range(len(objs)) if (im == k + 1).sum() >= 20]  # a fully (or almost) hidden object is not annotated
        if not keep:
            continue
        images[i], j = img, 0
        inst_i = np.zeros_like(im)
        for k in keep:
            ys, xs = np.nonzero(im == k + 1)
            inst_i[im == k + 1] = j + 1
            boxes[i, j] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            labels[i, j] = objs[k][0]
            for p, (x, y) in enumerate(objs[k][1]):
                kps[i, j, p] = (x, y)
                vis[i, j, p] = 2 if im[int(y), int(x)] == k + 1 else 1
            j += 1
        inst[i] = inst_i
        i += 1
    spec = {"synthetic": True, "generator": "examples/make_domain_fixtures.py:vision_fixture", "n": n, "size": size, "seed": seed, "classes": VISION_CLASSES,
            "keypointNames": KEYPOINT_NAMES, "flipPairs": FLIP_PAIRS, "boxFormat": "xyxy", "coords": "pixel", "colorSpace": "RGB", "channelOrder": "HWC uint8 0-255 (stored); CHW in the workbench",
            "boxConvention": "tight bounds of visible pixels; x_max/y_max exclusive pixel edges", "keypointConvention": "continuous pixel coordinates; visibility 2 visible, 1 occluded, 0 not labelled",
            "ids": [f"shape_{k:04d}" for k in range(n)]}
    return {"images": images, "instances": inst, "boxes": boxes, "labels": labels, "keypoints": kps, "visibility": vis, "spec": np.array(json.dumps(spec, sort_keys=True))}


# --------------------------------------------------------------------------------------------------------------------------------------- NLP
_SYL = ["bra", "ko", "vik", "sel", "mor", "tan", "ur", "li", "dax", "pen", "zor", "qui", "mel", "ando", "ber", "cal", "nor", "vu", "fen", "tis"]
_ORG_SUFFIX = ["Corp", "Labs", "Institute", "Works", "Group"]
_FILL = ["Yesterday", "Today", "Last week", "Reportedly"]
TEMPLATES = [
    "{PER} works at {ORG} in {LOC}.", "{PER} and {PER} visited {LOC} last week.", "The {ORG} office in {LOC} hired {PER}.", "{LOC} is cold in winter.",
    "{FILL} {PER} told {ORG} about the plan.", "{ORG} opened a branch near {LOC}, said {PER}.", "Nothing happened in the meeting today.", "{PER} left {LOC} for {LOC}.",
]


def _name(rng, n_syl: int) -> str:
    return "".join(rng.choice(_SYL) for _ in range(n_syl)).capitalize()


def _entity(rng, kind: str) -> str:
    if kind == "PER":
        return f"{_name(rng, int(rng.integers(1, 3)))} {_name(rng, int(rng.integers(2, 4)))}"
    if kind == "LOC":
        return f"{_name(rng, int(rng.integers(2, 4)))}" if rng.random() < 0.7 else f"Port-{_name(rng, 2)}"
    return f"{_name(rng, 2)} {rng.choice(_ORG_SUFFIX)}"


def ner_fixture(n: int = 240, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        tpl = TEMPLATES[int(rng.integers(0, len(TEMPLATES)))]
        text, spans, pos = "", [], 0
        while pos < len(tpl):
            if tpl[pos] == "{":
                end = tpl.index("}", pos)
                key = tpl[pos + 1:end]
                if key == "FILL":
                    text += str(rng.choice(_FILL))
                else:
                    ent = _entity(rng, key)
                    spans.append({"start": len(text), "end": len(text) + len(ent), "label": key})
                    text += ent
                pos = end + 1
            else:
                text += tpl[pos]
                pos += 1
        out.append({"id": f"ner_{i:04d}", "text": text, "spans": spans, "synthetic": True})
    return out


# --------------------------------------------------------------------------------------------------------------------------------------- speech
SR = 16000
TONES = {"a": 300.0, "b": 600.0, "c": 1000.0, "d": 1500.0, " ": 2200.0}
DURATIONS = (1.0, 1.5, 2.0)
SYMBOL_COUNT = {1.0: (3, 4), 1.5: (4, 6), 2.0: (5, 8)}  # letters per clip
MAX_SYMBOLS = {1.0: 5, 1.5: 8, 2.0: 11}  # letters + separators that fit the clip



def _transcript(rng, m: int) -> str:
    letters = [str(rng.choice(list("abcd"))) for _ in range(m)]
    # words separated by a single separator tone (never first, never last)
    text, run = "", 0
    for k, ch in enumerate(letters):
        if run >= 1 and k < m - 1 and rng.random() < 0.3:
            text += " "
            run = 0
        text += ch
        run += 1
    return text



def audio_fixture(n: int = 48, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    max_len = int(max(DURATIONS) * SR)
    wav = np.zeros((n, max_len), np.int16)
    lengths = np.zeros(n, np.int32)
    texts, segs = [], []
    for i in range(n):
        dur = 2.0 if i == 0 else float(DURATIONS[int(rng.integers(0, 3))])  # clip 0 is exactly the 2 s / 32,000-sample teaching length
        L = int(round(dur * SR))
        lo, hi = SYMBOL_COUNT[dur]
        text = _transcript(rng, int(rng.integers(lo, hi + 1)))
        while len(text) > MAX_SYMBOLS[dur]:
            text = text.replace(" ", "", 1)  # merge words until every symbol fits
        x = np.zeros(L, np.float64)
        t = int(rng.integers(0, int(0.06 * SR)))
        clip_segs = []
        for ch in text:
            d = int(rng.integers(int(0.09 * SR), int(0.12 * SR)))
            if t + d > L:
                raise RuntimeError("fixture symbol does not fit")
            tt = np.arange(d) / SR
            env = np.minimum(1, np.minimum(np.arange(d), d - 1 - np.arange(d)) / (0.008 * SR))
            x[t:t + d] += 0.6 * rng.uniform(0.8, 1.0) * np.sin(2 * np.pi * TONES[ch] * tt) * env
            clip_segs.append({"symbol": ch, "start": t, "end": t + d})
            t += d + int(rng.integers(int(0.025 * SR), int(0.045 * SR)))
        x += rng.normal(0, 0.01, L)
        wav[i, :L] = np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)
        lengths[i], _ = L, texts.append(text)
        segs.append(clip_segs)
    spec = {"synthetic": True, "generator": "examples/make_domain_fixtures.py:audio_fixture", "n": n, "seed": seed, "sampleRate": SR, "channels": 1, "tones": TONES,
            "alphabet": list(TONES), "durationsSeconds": list(DURATIONS), "dtype": "int16 PCM, scale 1/32767", "ids": [f"tone_{k:04d}" for k in range(n)],
            "teachingClip": "tone_0000 lasts exactly 2.0 s = 32,000 samples per channel at 16,000 Hz"}
    return {"waveforms": wav, "lengths": lengths, "transcripts": np.array(texts), "segments": np.array(json.dumps(segs)), "spec": np.array(json.dumps(spec, sort_keys=True))}


# --------------------------------------------------------------------------------------------------------------------------------------- files
def write_vision(path: Path = VISION_NPZ, **kw) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **vision_fixture(**kw))
    return path


def write_audio(path: Path = AUDIO_NPZ, **kw) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **audio_fixture(**kw))
    return path


def write_ner(path: Path = NER_JSONL, **kw) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in ner_fixture(**kw)))
    return path


def main(argv: list[str]) -> None:
    for p in (write_ner(), write_vision(), write_audio()):
        print(f"wrote {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main(sys.argv[1:])
