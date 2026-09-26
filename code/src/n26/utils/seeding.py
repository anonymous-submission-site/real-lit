"""Deterministic seed derivation: same cell tuple → same RNG seed."""
from __future__ import annotations
import hashlib


def derive_seed(
    dataset: str,
    target: str,
    regime: str,
    rate: int,
    seed: int,
    *extra: object,
) -> int:
    """Hash a cell-tuple to a stable 32-bit unsigned int.

    Use this to seed numpy.random.default_rng for masks, splits, etc.
    The hash is stable across machines and Python versions because
    SHA-256 is.
    """
    parts = [dataset, target, regime, str(rate), str(seed)]
    parts.extend(str(x) for x in extra)
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:4], "big")
