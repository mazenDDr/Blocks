"""Deterministically generate a tiny 10-class 64x64 RGB image-folder dataset.

    python examples/make_shapes10.py [--out examples/data/shapes10] [--per-class 20]

Each class is a shape (circle, square, ...) with seeded random colour, size and position.
The output is byte-identical for a given seed and Pillow version."""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 64
CLASSES = ["circle", "square", "triangle", "cross", "diamond", "ring", "hbar", "vbar", "star", "x"]


def _draw(shape: str, d: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color: tuple[int, int, int]) -> None:
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "triangle":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
    elif shape == "cross":
        t = max(2, r // 3)
        d.rectangle([cx - t, cy - r, cx + t, cy + r], fill=color)
        d.rectangle([cx - r, cy - t, cx + r, cy + t], fill=color)
    elif shape == "diamond":
        d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=color)
    elif shape == "ring":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=max(2, r // 3))
    elif shape == "hbar":
        d.rectangle([cx - r, cy - r // 3, cx + r, cy + r // 3], fill=color)
    elif shape == "vbar":
        d.rectangle([cx - r // 3, cy - r, cx + r // 3, cy + r], fill=color)
    elif shape == "star":
        pts = []
        for i in range(10):
            rad = r if i % 2 == 0 else r * 0.45
            a = -math.pi / 2 + i * math.pi / 5
            pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
        d.polygon(pts, fill=color)
    elif shape == "x":
        w = max(2, r // 3)
        d.line([(cx - r, cy - r), (cx + r, cy + r)], fill=color, width=w)
        d.line([(cx - r, cy + r), (cx + r, cy - r)], fill=color, width=w)
    else:
        raise ValueError(shape)


def generate(out: Path, per_class: int = 20, seed: int = 0) -> int:
    rng = random.Random(seed)
    n = 0
    for shape in CLASSES:
        (out / shape).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            bg = tuple(rng.randint(0, 70) for _ in range(3))
            color = tuple(rng.randint(130, 255) for _ in range(3))
            r = rng.randint(12, 20)
            cx, cy = rng.randint(r + 2, SIZE - r - 3), rng.randint(r + 2, SIZE - r - 3)
            img = Image.new("RGB", (SIZE, SIZE), bg)
            _draw(shape, ImageDraw.Draw(img), cx, cy, r, color)
            img.save(out / shape / f"{i:03d}.png")
            n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "shapes10"))
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(f"wrote {generate(Path(a.out), a.per_class, a.seed)} images to {a.out}")
