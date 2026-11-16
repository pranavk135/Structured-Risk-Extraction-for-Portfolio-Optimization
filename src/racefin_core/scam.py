"""SCAM math: Staleness, Conflict, Missingness, low-Authority (S, C, M, A) reliability model.

R(x) = clip(1 - sum_i w_i x_i - gamma * sum_{i<j} x_i x_j, 0, 1)

* w_i   : single-defect penalties (defaults match racefin_engine.score_evidence)
* gamma : extra penalty per *pair* of co-occurring defects  -> non-additive (compound) behaviour
Full factorial = 2^4 = 16 runs. Half fraction 2^(4-1) = 8 runs with generator A = S*C*M (+/-1 coding).
"""
from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pandas as pd

F = ["S", "C", "M", "A"]
NAMES = {"S": "Staleness", "C": "Conflict", "M": "Missingness", "A": "Low authority"}
DEFAULT_W = {"S": 0.15, "C": 0.25, "M": 0.20, "A": 0.25}


def reliability(x: dict, w: dict, gamma: float) -> float:
    base = sum(w[f] for f in F if x.get(f))
    pairs = sum(1 for a, b in combinations(F, 2) if x.get(a) and x.get(b))
    return float(np.clip(1 - base - gamma * pairs, 0, 1))


def penalty_breakdown(x: dict, w: dict, gamma: float) -> list[tuple[str, float]]:
    parts = [(NAMES[f], -w[f]) for f in F if x.get(f)]
    pairs = sum(1 for a, b in combinations(F, 2) if x.get(a) and x.get(b))
    if pairs and gamma:
        parts.append((f"Compound ({pairs} pair{'s' if pairs > 1 else ''})", -gamma * pairs))
    return parts


def full_factorial(w: dict, gamma: float) -> pd.DataFrame:
    rows = []
    for bits in product([0, 1], repeat=4):
        x = dict(zip(F, bits))
        rows.append({**x, "R": reliability({k: bool(v) for k, v in x.items()}, w, gamma)})
    df = pd.DataFrame(rows)
    pm = df[F] * 2 - 1
    df["half_fraction"] = pm["A"] == pm["S"] * pm["C"] * pm["M"]
    return df


def effects(df: pd.DataFrame) -> pd.Series:
    """Main + two-way effects = mean(R | +1) - mean(R | -1) under +/-1 coding."""
    pm = df[F] * 2 - 1
    out = {}
    for f in F:
        out[f] = df.loc[pm[f] == 1, "R"].mean() - df.loc[pm[f] == -1, "R"].mean()
    for a, b in combinations(F, 2):
        prod = pm[a] * pm[b]
        out[f"{a}×{b}"] = df.loc[prod == 1, "R"].mean() - df.loc[prod == -1, "R"].mean()
    return pd.Series(out)


ALIASES = {"S": "C×M×A", "C": "S×M×A", "M": "S×C×A", "A": "S×C×M",
           "S×C": "M×A", "S×M": "C×A", "S×A": "C×M"}  # half-fraction confounding (defining relation I = SCMA)
