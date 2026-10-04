"""Framing, windowing, STFT and mel features with the configuration DECLARED, and the frame-count formula derived from it (never guessed from duration alone).

    center=True  (padding P = n_fft // 2 on both sides): frames = 1 + floor((N + 2P - n_fft) / hop)
    center=False (no padding; only whole windows):                                    frames = 1 + floor((N - n_fft) / hop)   for N >= n_fft, else 0

Features come from torchaudio.transforms.MelSpectrogram (which uses torch.stft); the tests compare the frame count with the real STFT output."""
from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field


class FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_fft: int = Field(400, ge=16, le=4096, title="FFT size (samples)")
    win_length: int = Field(400, ge=1, le=4096, title="window length (samples, <= n_fft)")
    hop_length: int = Field(160, ge=1, le=4096, title="hop (samples)")
    window: Literal["hann", "hamming"] = "hann"
    center: bool = Field(True, title="pad n_fft//2 on both sides so frame t is centred at t*hop")
    pad_mode: Literal["reflect", "constant"] = "reflect"
    n_mels: int = Field(40, ge=4, le=256)
    f_min: float = 0.0
    f_max: float | None = Field(None, title="highest mel frequency (Hz; empty = sample_rate / 2)")
    power: Literal[1, 2] = Field(2, title="1 = magnitude, 2 = power")
    log: bool = Field(True, title="natural log of (mel + floor)")
    floor: float = Field(1e-6, gt=0)


def frame_count(n_samples: int, n_fft: int, hop: int, center: bool) -> int:
    if center:
        return max(0, 1 + (n_samples + 2 * (n_fft // 2) - n_fft) // hop)
    return 1 + (n_samples - n_fft) // hop if n_samples >= n_fft else 0


def frame_time(t: int, hop: int, sample_rate: int, center: bool, n_fft: int) -> tuple[float, float]:
    """(start, centre) time in seconds of frame t of the original signal. With center=True frame t is centred on sample t*hop."""
    start = t * hop - (n_fft // 2 if center else 0)
    return start / sample_rate, (start + n_fft / 2) / sample_rate


def window_fn(name: str):
    return {"hann": torch.hann_window, "hamming": torch.hamming_window}[name]


def check_config(cfg: FeatureConfig, sample_rate: int) -> list[tuple[str, str]]:
    bad = []
    if cfg.win_length > cfg.n_fft:
        bad.append(("E_AUDIO_FRAME_CONFIG", f"win_length ({cfg.win_length}) must not exceed n_fft ({cfg.n_fft})"))
    if cfg.f_min < 0 or cfg.f_min >= sample_rate / 2:
        bad.append(("E_AUDIO_FRAME_CONFIG", "f_min must be nonnegative and below Nyquist"))
    if cfg.f_max is not None and cfg.f_max > sample_rate / 2:
        bad.append(("E_AUDIO_FRAME_CONFIG", f"f_max {cfg.f_max} Hz is above the Nyquist frequency {sample_rate / 2} Hz"))
    if cfg.f_max is not None and cfg.f_max <= cfg.f_min:
        bad.append(("E_AUDIO_FRAME_CONFIG", "f_max must be above f_min"))
    return bad


def mel_transform(cfg: FeatureConfig, sample_rate: int):
    import warnings

    import torchaudio.transforms as AT

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AT.MelSpectrogram(sample_rate=sample_rate, n_fft=cfg.n_fft, win_length=cfg.win_length, hop_length=cfg.hop_length, f_min=cfg.f_min, f_max=cfg.f_max,
                                 n_mels=cfg.n_mels, window_fn=window_fn(cfg.window), power=float(cfg.power), center=cfg.center, pad_mode=cfg.pad_mode, norm=None, mel_scale="htk")


def features_for(wave: torch.Tensor, cfg: FeatureConfig, sample_rate: int) -> torch.Tensor:
    """(channels, samples) -> (frames, n_mels). Channels are mixed to mono by the mean (declared by the caller)."""
    mono = wave.mean(0, keepdim=True)
    m = mel_transform(cfg, sample_rate)(mono)[0]  # (n_mels, frames)
    if cfg.log:
        m = torch.log(m + cfg.floor)
    return m.T.contiguous()


def pad_features(feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (B, T_max, n_mels) zero padded, lengths (B,), boolean mask (B, T_max) that is True on real frames."""
    lengths = torch.tensor([f.shape[0] for f in feats])
    T = int(lengths.max())
    x = torch.zeros(len(feats), T, feats[0].shape[1])
    for i, f in enumerate(feats):
        x[i, :f.shape[0]] = f
    mask = torch.arange(T)[None, :] < lengths[:, None]
    return x, lengths, mask


def conv_out_len(t: torch.Tensor | int, kernel: int, stride: int, padding: int):
    return (t + 2 * padding - kernel) // stride + 1


def describe(cfg: FeatureConfig, sample_rate: int) -> dict[str, Any]:
    return {**cfg.model_dump(), "sampleRate": sample_rate, "frameSeconds": cfg.hop_length / sample_rate, "windowSeconds": cfg.win_length / sample_rate,
            "freqBins": cfg.n_fft // 2 + 1,
            "formula": ("frames = 1 + floor((N + 2*floor(n_fft/2) - n_fft) / hop)  [center: pad_mode=" + cfg.pad_mode + "; even n_fft simplifies to 1 + floor(N/hop)]") if cfg.center
            else "frames = 1 + floor((N - n_fft) / hop)  [no padding; 0 when N < n_fft]"}
