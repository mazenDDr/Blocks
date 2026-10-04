"""Write the deterministic SYNTHETIC CSV fixtures used by the tabular examples and tests.

    python examples/make_tabular_fixtures.py        # writes examples/fixtures/*.csv

Everything here is generated from fixed seeds; none of it is measured data.
- synthetic_housing.csv        regression journey (VISION 11): price from area, rooms, age, neighbourhood + noise, some missing values, some exact duplicates
- synthetic_churn.csv          binary classification (logistic regression)
- synthetic_enzyme_two_group.csv   independent two-group comparison: enzyme activity of control vs inhibitor-treated tubes
- synthetic_enzyme_paired.csv  paired comparison: the same specimens before and after treatment
- synthetic_cells.csv          unsupervised (Milestone 5): three generating groups of "cells" with features on very different scales; `true_group` is the generating label
- synthetic_moons.csv          unsupervised (Milestone 5): two interleaving half-moons (non-convex clusters); `true_shape` is the generating label
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "fixtures"


def housing() -> pd.DataFrame:
    rng = np.random.default_rng(1234)
    n = 400
    area = rng.uniform(35, 160, n).round(1)
    rooms = np.clip(np.round(area / 32 + rng.normal(0, 0.6, n)), 1, 6).astype(int)
    age = rng.uniform(0, 60, n).round(1)
    hood = rng.choice(["north", "central", "south"], n, p=[0.3, 0.4, 0.3])
    bump = pd.Series(hood).map({"north": -12.0, "central": 30.0, "south": 0.0}).to_numpy()
    price = (25 + 2.2 * area + 9 * rooms - 0.8 * age + bump + rng.normal(0, 14, n)).round(1)
    df = pd.DataFrame({"id": np.arange(1, n + 1), "area_m2": area, "rooms": rooms, "age_years": age, "neighborhood": hood, "price_k": price})
    df.loc[rng.choice(n, 22, replace=False), "area_m2"] = np.nan
    df.loc[rng.choice(n, 18, replace=False), "age_years"] = np.nan
    dups = df.iloc[rng.choice(n, 12, replace=False)]
    return pd.concat([df, dups], ignore_index=True).sample(frac=1.0, random_state=7).reset_index(drop=True)


def churn() -> pd.DataFrame:
    rng = np.random.default_rng(99)
    n = 300
    tenure = rng.uniform(1, 72, n).round(0)
    monthly = rng.uniform(20, 120, n).round(1)
    tickets = rng.poisson(1.5, n)
    z = -0.5 - 0.05 * tenure + 0.03 * monthly + 0.35 * tickets
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame({"customer": np.arange(1, n + 1), "tenure_months": tenure, "monthly_charge": monthly, "support_tickets": tickets, "churned": y})


def enzyme_two_group() -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    ctrl = rng.normal(50.0, 4.0, 12).round(2)
    inhib = rng.normal(43.0, 7.5, 14).round(2)
    rows = [(f"T{i + 1:02d}", "control", v, "A" if i % 2 == 0 else "B") for i, v in enumerate(ctrl)]
    rows += [(f"T{i + 13:02d}", "inhibitor", v, "A" if i % 2 == 0 else "B") for i, v in enumerate(inhib)]
    return pd.DataFrame(rows, columns=["tube_id", "condition", "activity_u_per_mg", "batch"])


def enzyme_paired() -> pd.DataFrame:
    rng = np.random.default_rng(77)
    base = rng.normal(48, 6, 10)
    post = base - rng.normal(4.0, 2.0, 10)
    rows = []
    for i in range(10):
        rows.append((f"S{i + 1:02d}", "before", round(float(base[i]), 2)))
        rows.append((f"S{i + 1:02d}", "after", round(float(post[i]), 2)))
    return pd.DataFrame(rows, columns=["specimen", "timepoint", "activity_u_per_mg"])


def cells() -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    spec = [("A", 140, [90.0, 0.30, 12.0, 1.1], [8.0, 0.05, 2.0, 0.08]), ("B", 140, [140.0, 0.55, 20.0, 1.4], [10.0, 0.06, 2.5, 0.10]), ("C", 80, [148.0, 0.60, 24.0, 1.62], [12.0, 0.07, 3.0, 0.12])]
    rows = []
    for g, n, mu, sd in spec:
        x = rng.normal(mu, sd, size=(n, 4))
        rows.append(pd.DataFrame({"area_um2": x[:, 0].round(2), "intensity": x[:, 1].round(4), "granularity": x[:, 2].round(3), "elongation": x[:, 3].round(4), "true_group": g}))
    df = pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=5).reset_index(drop=True)
    df.insert(0, "cell_id", np.arange(1, len(df) + 1))
    return df


def moons() -> pd.DataFrame:
    rng = np.random.default_rng(77)
    n = 150
    t = rng.uniform(0, np.pi, n)
    a = np.c_[np.cos(t), np.sin(t)]
    b = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
    pts = np.r_[a, b] + rng.normal(0, 0.06, (2 * n, 2))
    df = pd.DataFrame({"x": pts[:, 0].round(4), "y": pts[:, 1].round(4), "true_shape": ["upper"] * n + ["lower"] * n})
    df = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    df.insert(0, "point_id", np.arange(1, len(df) + 1))
    return df


FIXTURES = {"synthetic_housing.csv": housing, "synthetic_churn.csv": churn, "synthetic_enzyme_two_group.csv": enzyme_two_group,
            "synthetic_enzyme_paired.csv": enzyme_paired, "synthetic_cells.csv": cells, "synthetic_moons.csv": moons}


def main(out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIXTURES.items():
        fn().to_csv(out / name, index=False)
        print("wrote", out / name)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
