# Copyright (c) 2026 Tanmay Roy. All rights reserved.
# Proprietary and view-only; copying, modification or reuse is prohibited. See LICENSE.

"""Judge calibration: agreement between LLM-judge scores and human labels.

A judge is a measurement instrument; before its scores gate a release we check it agrees
with humans on a labelled sample. Metrics:

* **Quadratic-weighted Cohen's kappa** — chance-corrected agreement on ordinal 1-5 scores
  (near-misses penalised less than large disagreements). Rule of thumb: ≥0.6 substantial.
* **Spearman rank correlation** — does the judge order answers like humans do?
* **Exact / within-1 agreement** — easy to explain to stakeholders.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def quadratic_weighted_kappa(a: list[int], b: list[int], min_r: int = 1, max_r: int = 5) -> float:
    n = max_r - min_r + 1
    obs = [[0.0] * n for _ in range(n)]
    for x, y in zip(a, b, strict=True):
        obs[x - min_r][y - min_r] += 1
    total = len(a)
    hist_a = [sum(row) for row in obs]
    hist_b = [sum(obs[i][j] for i in range(n)) for j in range(n)]
    num = den = 0.0
    for i in range(n):
        for j in range(n):
            w = ((i - j) ** 2) / ((n - 1) ** 2)
            expected = hist_a[i] * hist_b[j] / total
            num += w * obs[i][j]
            den += w * expected
    return 1.0 - num / den if den else 1.0


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = _ranks(a), _ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def calibrate(path: Path, kappa_threshold: float = 0.6) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    human = [int(r["human_score"]) for r in rows]
    judge = [round(float(r["judge_score"])) for r in rows]
    kappa = quadratic_weighted_kappa(human, judge)
    return {
        "n": len(rows),
        "quadratic_weighted_kappa": round(kappa, 3),
        "spearman": round(spearman([float(x) for x in human], [float(x) for x in judge]), 3),
        "exact_agreement": round(sum(h == j for h, j in zip(human, judge, strict=True)) / len(rows), 3),
        "within_one": round(sum(abs(h - j) <= 1 for h, j in zip(human, judge, strict=True)) / len(rows), 3),
        "mean_bias": round(sum(j - h for h, j in zip(human, judge, strict=True)) / len(rows), 3),
        "trusted": kappa >= kappa_threshold and len(rows) >= 50,
        "note": "Judge is trusted for gating only with n>=50 labelled items and kappa>=threshold.",
    }
