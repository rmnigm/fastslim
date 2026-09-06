from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from . import api

__all__ = ["NotFittedError", "SLIM"]

PARAM_NAMES = ("lambd", "beta", "max_iter", "tol", "n_threads")
DEFAULTS: dict[str, Any] = {
    "lambd": 0.5,
    "beta": 0.5,
    "max_iter": 1000,
    "tol": 1e-4,
    "n_threads": None,
}


class NotFittedError(ValueError, AttributeError):
    """Raised when a :class:`SLIM` instance is used before ``fit``.

    Subclasses both :class:`ValueError` and :class:`AttributeError`.
    """


class SLIM:
    """SLIM item-item recommender: a thin wrapper around :func:`fastslim.fit`.

    ``fit`` sets ``weights``, ``n_items``, ``n_passes`` and ``converged``;
    hyperparameters and methods are documented in `docs/api.md`.
    """

    def __init__(
        self,
        lambd: float = 0.5,
        beta: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-4,
        n_threads: int | None = None,
    ) -> None:
        self.lambd = lambd
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.n_threads = n_threads

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the hyperparameters as a dict; ``deep`` is accepted and ignored."""
        del deep
        return {name: getattr(self, name) for name in PARAM_NAMES}

    def set_params(self, **params: Any) -> SLIM:
        """Set hyperparameters and return ``self``."""
        for name, value in params.items():
            if name not in PARAM_NAMES:
                raise ValueError(
                    f"invalid parameter {name!r} for SLIM; "
                    f"valid parameters are {', '.join(PARAM_NAMES)}"
                )
            setattr(self, name, value)
        return self

    # X/X_history keep the scikit-learn spelling; renaming breaks callers.
    def fit(self, X: Any, y: Any = None) -> SLIM:  # noqa: N803
        """Fit the item-item weights on a user-item interaction matrix.

        Returns ``self``; ``y`` is ignored.
        """
        del y
        self.weights, self.n_passes, self.converged = api.fit_with_diagnostics(
            X, self.lambd, self.beta, self.max_iter, self.tol, self.n_threads
        )
        self.n_items = int(self.weights.shape[0])
        return self

    def fitted_weights(self) -> sparse.csr_matrix:
        """Return the fitted ``weights``, or raise :class:`NotFittedError`."""
        weights = getattr(self, "weights", None)
        if weights is None:
            raise NotFittedError(
                "this SLIM instance is not fitted yet; call fit() with an "
                "interaction matrix before using this method"
            )
        return weights

    def predict(
        self,
        X_history: Any,  # noqa: N803
        *,
        exclude_seen: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Score every item for one user or a batch; see :func:`fastslim.predict`."""
        return api.predict(
            self.fitted_weights(),
            X_history,
            exclude_seen=exclude_seen,
            batch_size=batch_size,
        )

    def recommend(
        self,
        X_history: Any,  # noqa: N803
        k: int = 10,
        *,
        exclude_seen: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Top-``k`` items for one user or a batch; see :func:`fastslim.recommend`."""
        return api.recommend(
            self.fitted_weights(),
            X_history,
            k=k,
            exclude_seen=exclude_seen,
            batch_size=batch_size,
        )

    def __repr__(self) -> str:
        """Show only the hyperparameters that differ from the defaults."""
        changed = [
            f"{name}={getattr(self, name)!r}"
            for name in PARAM_NAMES
            if getattr(self, name) != DEFAULTS[name]
        ]
        return f"SLIM({', '.join(changed)})"
