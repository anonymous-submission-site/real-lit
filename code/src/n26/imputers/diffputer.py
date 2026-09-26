"""DiffPuter — EM-driven latent-space diffusion imputer scaffold. Reference: Zhang et al. (2025)."""
from __future__ import annotations

import numpy as np

from n26.imputers import register_imputer
from n26.imputers.base import Imputer


@register_imputer("DiffPuter")
class DiffPuterImputer(Imputer):
    """DiffPuter wrapper. Requires the public DiffPuter reference implementation."""

    def __init__(self, **_: object) -> None:
        pass

    def fit_transform(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "DiffPuter is a scaffold in the public harness (one of the K-sweep "
            "learned imputers in the paper's Method section). Obtain the public "
            "reference implementation from Zhang et al. 2025 and replace "
            "this stub's .fit_transform with the EM-driven diffusion "
            "imputer fit on the training fold's observed entries only."
        )
