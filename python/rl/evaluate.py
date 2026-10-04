"""Evaluation protocol (VISION 9.9 'Evaluation and scientific control'): SEPARATE environment instances, declared seeds, greedy policy.

Each evaluation episode gets its own freshly built environment seeded with one of the declared evaluation seeds, so every variant, run and
policy version is evaluated on the identical initial conditions. Reported separately: the return under the TRAINING reward definition
(weights as trained) and under the environment's DEFAULT reward (the task return, comparable across reward variants), per-component
returns, episode length, termination vs truncation counts and the catalog's success rule."""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import stats

from .envs import CATALOG, EnvSpec, RewardSpec, make_env


def frame_png(frame: np.ndarray, max_width: int = 320) -> bytes:
    im = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    if im.width > max_width:
        h = max(1, round(im.height * max_width / im.width))
        im = im.resize((max_width, h), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def mean_ci(values: list[float], level: float = 0.95) -> dict[str, Any]:
    v = np.asarray(values, float)
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci": None}
    mean = float(v.mean())
    if n < 2:
        return {"n": n, "mean": mean, "std": None, "ci": None, "level": level, "note": "one sample: no interval"}
    sd = float(v.std(ddof=1))
    h = float(stats.t.ppf(0.5 + level / 2, n - 1) * sd / np.sqrt(n))
    return {"n": n, "mean": mean, "std": sd, "ci": [mean - h, mean + h], "level": level, "method": "Student t interval of the mean"}


def success_of(env_id: str, terminated: bool, truncated: bool) -> bool | None:
    rule = CATALOG[env_id]["successRule"]
    return None if rule is None else (terminated if rule == "terminated" else truncated and not terminated)


def evaluate_policy(q_net, spec: EnvSpec, reward: RewardSpec, seeds: list[int], *, epsilon: float = 0.0, frames_for: int = 0, frame_width: int = 320, max_frames: int = 300,
                    max_steps: int | None = None) -> dict[str, Any]:
    """Run one greedy episode per seed on a new environment instance. `frames_for` = number of leading episodes to render (real env.render() frames)."""
    cat = CATALOG[spec.env_id]
    comps = cat["rewardComponents"]
    rows, captured = [], []
    rng = np.random.default_rng(12345)
    for j, s in enumerate(seeds):
        env = make_env(spec, reward, render=j < frames_for)
        try:
            obs, _ = env.reset(seed=int(s))
            frames = [frame_png(env.render(), frame_width)] if j < frames_for else None
            steps, ret, task, term, trunc = 0, 0.0, 0.0, False, False
            wsum, rsum = np.zeros(len(comps)), np.zeros(len(comps))
            qs, acts = [], []
            while not (term or trunc):
                with torch.no_grad():
                    q = q_net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0].numpy()
                a = int(rng.integers(len(q))) if epsilon > 0 and rng.random() < epsilon else int(q.argmax())
                obs, r, term, trunc, info = env.step(a)
                steps += 1
                ret += r
                task += info["native_reward"]
                wsum += [info["reward_components"][k] for k in comps]
                rsum += [info["reward_components_raw"][k] for k in comps]
                if frames is not None and len(frames) < max_frames:
                    frames.append(frame_png(env.render(), frame_width))
                    qs.append(q.tolist())
                    acts.append(a)
            rows.append({"seed": int(s), "length": steps, "return": ret, "taskReturn": task, "terminated": bool(term), "truncated": bool(trunc), "success": success_of(spec.env_id, term, trunc),
                         "components": dict(zip(comps, wsum.tolist())), "rawComponents": dict(zip(comps, rsum.tolist()))})
            if frames is not None:
                captured.append({"seed": int(s), "frames": frames, "qValues": qs, "actions": acts, "length": steps, "taskReturn": task, "terminated": bool(term), "truncated": bool(trunc),
                                 "framesCapped": steps + 1 > max_frames})
        finally:
            env.close()
    succ = [r["success"] for r in rows if r["success"] is not None]
    report = {
        "seeds": [int(s) for s in seeds], "episodes": rows, "epsilon": epsilon,
        "return": mean_ci([r["return"] for r in rows]), "taskReturn": mean_ci([r["taskReturn"] for r in rows]), "length": mean_ci([r["length"] for r in rows]),
        "terminatedCount": sum(r["terminated"] for r in rows), "truncatedCount": sum(r["truncated"] for r in rows),
        "successRate": (sum(succ) / len(succ)) if succ else None, "successRule": cat["successText"],
        "componentReturns": {k: mean_ci([r["components"][k] for r in rows]) for k in comps},
        "rawComponentReturns": {k: mean_ci([r["rawComponents"][k] for r in rows]) for k in comps},
        "rewardBasis": "return = sum of the reward the learner trained on (weights as configured); taskReturn = sum of the environment's DEFAULT reward, comparable across reward variants",
    }
    return {"report": report, "captured": captured}
