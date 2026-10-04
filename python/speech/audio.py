"""Audio contract: waveform samples, sample rate, channels, duration, and the connection between a sample index and time (t = i / sample_rate).

The teaching signal (VISION 9.8): a 2 s input at 16,000 samples/s has exactly 32,000 samples per channel. `teaching_signal()` builds it (a 440 Hz sine, labelled as a teaching signal)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

TEACHING_SECONDS, TEACHING_RATE = 2.0, 16000
TEACHING_SAMPLES_PER_CHANNEL = 32000


@dataclass
class Clip:
    id: str
    wave: torch.Tensor  # (channels, samples) float32 in [-1, 1]
    sample_rate: int
    text: str | None = None
    segments: list[dict] | None = None  # true symbol segments (start/end in samples) of the SYNTHETIC fixture

    @property
    def channels(self) -> int:
        return int(self.wave.shape[0])

    @property
    def samples_per_channel(self) -> int:
        return int(self.wave.shape[1])

    @property
    def duration(self) -> float:
        return self.samples_per_channel / self.sample_rate


def teaching_signal(seconds: float = TEACHING_SECONDS, sample_rate: int = TEACHING_RATE, hz: float = 440.0) -> Clip:
    n = int(round(seconds * sample_rate))
    t = torch.arange(n, dtype=torch.float64) / sample_rate
    return Clip("teaching_signal", (0.5 * torch.sin(2 * torch.pi * hz * t)).to(torch.float32).unsqueeze(0), sample_rate)


def time_of_sample(i: int, sample_rate: int) -> float:
    return i / sample_rate


def sample_of_time(t: float, sample_rate: int) -> int:
    return int(round(t * sample_rate))


def contract(c: Clip) -> dict[str, Any]:
    return {"id": c.id, "sampleRate": c.sample_rate, "channels": c.channels, "samplesPerChannel": c.samples_per_channel, "durationSeconds": c.duration,
            "totalSamples": c.channels * c.samples_per_channel, "peak": float(c.wave.abs().max()), "clippedSamples": int((c.wave.abs() >= 0.999).sum()),
            "silentFraction": float((c.wave.abs() < 0.02).float().mean())}


def load_npz(path, n: int | None) -> tuple[list[Clip], dict[str, Any]]:
    d = np.load(path, allow_pickle=False)
    spec = json.loads(str(d["spec"]))
    segs = json.loads(str(d["segments"]))
    k = len(d["lengths"]) if n is None else min(n, len(d["lengths"]))
    clips = []
    for i in range(k):
        L = int(d["lengths"][i])
        if d["waveforms"].ndim == 3:
            w = torch.from_numpy(d["waveforms"][i, :, :L].astype(np.float32))
        else:
            w = torch.from_numpy(d["waveforms"][i, :L].astype(np.float32) / 32767.0).unsqueeze(0)
        clips.append(Clip(spec["ids"][i], w, int(spec["sampleRate"]), str(d["transcripts"][i]), segs[i]))
    return clips, spec


def resample(c: Clip, target: int) -> Clip:
    import torchaudio.functional as AF

    if target == c.sample_rate:
        return c
    return Clip(c.id, AF.resample(c.wave, c.sample_rate, target), target, c.text,
                None if c.segments is None else [{**s, "start": int(round(s["start"] * target / c.sample_rate)), "end": int(round(s["end"] * target / c.sample_rate))} for s in c.segments])
