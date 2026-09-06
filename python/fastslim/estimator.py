"""The :class:`SLIM` estimator.

A thin, scikit-learn-flavoured wrapper around the functional API.  It follows
the usual conventions -- constructor stores hyperparameters untouched,
validation happens in ``fit``, learned state lives in trailing-underscore
attributes -- without importing scikit-learn.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from . import _api

__all__ = ["NotFittedError", "SLIM"]

_PARAM_NAMES = ("lambd", "beta", "max_iter", "tol", "n_threads")
_DEFAULTS: dict[str, Any] = {
    "lambd": 0.5,
    "beta": 0.5,
    "max_iter": 1000,
    "tol": 1e-4,
    "n_threads": None,
}


class NotFittedError(ValueError, AttributeError):
    """Raised when a :class:`SLIM` instance is used before ``fit``.

    Inherits from both :class:`ValueError` and :class:`AttributeError` so that
    ``except ValueError`` and ``except AttributeError`` both catch it, matching
    scikit-learn's exception of the same name.
    """


class SLIM:
    r"""SLIM item-item recommender.

    Learns a sparse, non-negative item-item weight matrix ``W`` by solving, for
    each item :math:`i`,

    .. math::

        \min_{w \ge 0,\, w_i = 0} \;
        \tfrac{1}{2} \lVert x_i - X w \rVert_2^2
        + \lambda \sum_k w_k + \tfrac{\beta}{2} \sum_k w_k^2 .

    Scores for a user are then ``history @ W``.

    Parameters
    ----------
    lambd : float, default=0.5
        L1 penalty; larger values give sparser ``W``.  Measured on the scale of
        the Gram matrix :math:`X^\top X`, i.e. co-occurrence counts for binary
        data: a neighbour :math:`k` of item :math:`i` can only enter the model
        when :math:`P_{ik} \ge \lambda`.
    beta : float, default=0.5
        L2 penalty; shrinks weights and makes the solution unique.  It also
        conditions the per-coordinate denominator :math:`P_{kk} + \beta`, so
        raising it cuts the number of passes a fit needs.
    max_iter : int, default=1000
        Maximum coordinate-descent passes per item; full passes and active-set
        passes both count towards it.
    tol : float, default=1e-4
        Convergence tolerance on the largest weight change within a pass.  An
        item is done once a full pass moves no weight by ``tol`` or more, so
        ``tol=0`` means exactly ``max_iter`` passes.
    n_threads : int or None, default=None
        Worker threads; ``None`` uses every available core.  Results do not
        depend on this value.

    Attributes
    ----------
    weights_ : scipy.sparse.csr_matrix
        Item-item weights, shape ``(n_items, n_items)``, available after
        ``fit``.
    n_items_ : int
        Number of items seen during ``fit``.
    n_passes_ : numpy.ndarray
        ``int64`` array of shape ``(n_items,)``: coordinate-descent passes
        actually used for each item (full and active-set passes both count).
    converged_ : numpy.ndarray
        ``bool`` array of shape ``(n_items,)``: whether each item reached
        ``tol`` before running out of ``max_iter``.  Items with no candidate
        neighbours count as converged in zero passes.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse
    >>> from fastslim import SLIM
    >>> X = sparse.csr_matrix(np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]]))
    >>> model = SLIM(lambd=0.1, beta=0.1).fit(X)
    >>> model.n_items_
    3
    >>> model.recommend(X[0], k=1).shape
    (1,)
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
        """Return the hyperparameters as a dict.

        Parameters
        ----------
        deep : bool, default=True
            Accepted for scikit-learn compatibility.  ``SLIM`` holds no nested
            estimators, so it makes no difference.
        """
        del deep
        return {name: getattr(self, name) for name in _PARAM_NAMES}

    def set_params(self, **params: Any) -> SLIM:
        """Set hyperparameters and return ``self``."""
        for name, value in params.items():
            if name not in _PARAM_NAMES:
                raise ValueError(
                    f"invalid parameter {name!r} for SLIM; "
                    f"valid parameters are {', '.join(_PARAM_NAMES)}"
                )
            setattr(self, name, value)
        return self

    def fit(self, X: Any, y: Any = None) -> SLIM:
        """Fit the item-item weights on a user-item interaction matrix.

        Parameters
        ----------
        X : sparse matrix, sparse array or array-like
            User-item interactions, shape ``(n_users, n_items)``.  Values must
            be finite and non-negative.
        y : ignored
            Present for API consistency; SLIM is unsupervised.

        Returns
        -------
        self : SLIM

        Warns
        -----
        ConvergenceWarning
            If some items exhausted ``max_iter`` before reaching ``tol``; see
            :attr:`converged_` for which ones.
        """
        del y
        self.weights_, self.n_passes_, self.converged_ = _api._fit_impl(
            X, self.lambd, self.beta, self.max_iter, self.tol, self.n_threads
        )
        self.n_items_ = int(self.weights_.shape[0])
        return self

    def _weights(self) -> sparse.csr_matrix:
        weights = getattr(self, "weights_", None)
        if weights is None:
            raise NotFittedError(
                "this SLIM instance is not fitted yet; call fit() with an "
                "interaction matrix before using this method"
            )
        return weights

    def predict(
        self,
        X_history: Any,
        *,
        exclude_seen: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Score every item for one user or a batch; see :func:`fastslim.predict`."""
        return _api.predict(
            self._weights(),
            X_history,
            exclude_seen=exclude_seen,
            batch_size=batch_size,
        )

    def recommend(
        self,
        X_history: Any,
        k: int = 10,
        *,
        exclude_seen: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Top-``k`` items for one user or a batch; see :func:`fastslim.recommend`."""
        return _api.recommend(
            self._weights(),
            X_history,
            k=k,
            exclude_seen=exclude_seen,
            batch_size=batch_size,
        )

    def __repr__(self) -> str:
        """Show only the hyperparameters that differ from the defaults.

        The result is valid Python that rebuilds an equivalent (unfitted)
        estimator, as scikit-learn's reprs are.
        """
        changed = [
            f"{name}={getattr(self, name)!r}"
            for name in _PARAM_NAMES
            if getattr(self, name) != _DEFAULTS[name]
        ]
        return f"SLIM({', '.join(changed)})"
