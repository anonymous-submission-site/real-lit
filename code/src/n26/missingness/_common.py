"""Shared mask-sampling utilities for MAR/MNAR (Schouten 2018, Muzellec et al. 2020)."""
from __future__ import annotations

import numpy as np

_TIE_NOISE_SCALE = 0.5


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise sigmoid."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _calibrate_intercept(
    logits: np.ndarray,
    eligible: np.ndarray,
    target_rate: float,
    tol: float = 1e-4,
    max_iter: int = 64,
) -> float:
    """Binary-search scalar b so that mean(sigmoid(logits[eligible] + b)) ≈ target_rate."""
    if not eligible.any():
        return 0.0
    elig_logits = logits[eligible]
    lo, hi = -30.0, 30.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        rate_now = float(_sigmoid(elig_logits + mid).mean())
        if abs(rate_now - target_rate) < tol:
            return mid
        if rate_now > target_rate:
            hi = mid
        else:
            lo = mid
    return mid


def _logistic_bernoulli_mask(
    logits: np.ndarray,        # (n, d) — unscaled logits per cell
    eligible: np.ndarray,      # (n, d) — True where cell can be masked (observed)
    rate: int,
    seed: int,
) -> np.ndarray:
    """Cell-wise Bernoulli mask with per-column intercept calibrated to the target rate."""
    if not (0 <= rate < 100):
        raise ValueError(f"rate must be in [0, 100), got {rate}")
    if rate == 0:
        return np.zeros_like(eligible, dtype=bool)
    rng = np.random.default_rng(seed)
    target_p = rate / 100.0
    n, d = logits.shape
    mask = np.zeros((n, d), dtype=bool)
    for j in range(d):
        col_eligible = eligible[:, j]
        if not col_eligible.any():
            continue
        b = _calibrate_intercept(logits[:, j], col_eligible, target_p)
        probs = _sigmoid(logits[:, j] + b)
        draws = rng.random(n) < probs
        mask[:, j] = draws & col_eligible
    return mask


def _topk_mask_from_scores(
    scores: np.ndarray,
    observed: np.ndarray,
    rate: int,
    seed: int,
) -> np.ndarray:
    """Mask the top `rate%` of observed cells by score (seeded tie-breaker noise)."""
    if not (0 <= rate < 100):
        raise ValueError(f"rate must be in [0, 100), got {rate}")
    if rate == 0:
        return np.zeros_like(observed, dtype=bool)
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=_TIE_NOISE_SCALE, size=scores.shape)
    safe_scores = np.where(np.isnan(scores), -np.inf, scores) + noise
    candidate_scores = np.where(observed, safe_scores, -np.inf)

    n_observed = int(observed.sum())
    k = int(round(rate / 100.0 * n_observed))
    if k == 0:
        return np.zeros_like(observed, dtype=bool)
    flat = candidate_scores.ravel()
    threshold_idx = -k
    threshold = np.partition(flat, threshold_idx)[threshold_idx]
    mask = candidate_scores >= threshold
    if mask.sum() > k:
        sorted_idx = np.argsort(-candidate_scores.ravel())[:k]
        mask = np.zeros_like(observed, dtype=bool)
        mask.ravel()[sorted_idx] = True
    return mask & observed
