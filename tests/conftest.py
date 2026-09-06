"""Shared fixtures and a pure-numpy reference implementation of the solver.

The reference solver, :func:`objective` and :func:`kkt_violation` all describe
the *same* optimisation problem the Rust code claims to solve, independently of
it.  Test modules import them directly (``from conftest import kkt_violation``);
pytest puts this directory on ``sys.path``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

# Objective, for every target item i, over w >= 0 with w_i = 0:
#
#     0.5 * ||x_i - X w||^2 + lambd * sum_k w_k + (beta / 2) * sum_k w_k^2
#
# Written with P = X.T @ X this is
#     0.5 * P_ii - w^T P_{:,i} + 0.5 * w^T P w + lambd * 1^T w + 0.5 * beta * w^T w
# whose coordinate-wise minimiser is the soft-thresholded ratio used below.


def to_dense(matrix) -> np.ndarray:
    """Dense float64 view of a sparse or dense matrix."""
    if sparse.issparse(matrix):
        return np.asarray(matrix.todense(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def reference_cd(
    X: np.ndarray,
    lambd: float,
    beta: float,
    max_passes: int = 5000,
    tol: float = 1e-13,
) -> np.ndarray:
    """Solve the SLIM problem by textbook non-negative coordinate descent.

    Deliberately naive: dense Gram matrix, full passes over every coordinate,
    no active set, no shortcuts.  Returns ``W`` with ``W[k, i] = w_ik``, so
    ``scores = X @ W`` exactly as :func:`fastslim.fit` returns it.
    """
    X = np.asarray(X, dtype=np.float64)
    P = X.T @ X
    n_items = P.shape[0]
    W = np.zeros((n_items, n_items), dtype=np.float64)
    for i in range(n_items):
        w = np.zeros(n_items)
        for _ in range(max_passes):
            max_delta = 0.0
            for k in range(n_items):
                if k == i:
                    continue
                # Partial residual correlation with coordinate k held out.
                c = P[i, k] - (P[k] @ w) + P[k, k] * w[k]
                denominator = P[k, k] + beta
                if denominator == 0.0:
                    continue
                w_new = max(0.0, (c - lambd) / denominator)
                delta = w_new - w[k]
                if delta != 0.0:
                    w[k] = w_new
                    max_delta = max(max_delta, abs(delta))
            if max_delta < tol:
                break
        W[:, i] = w
    return W


def objective(X, W, lambd: float, beta: float) -> float:
    """Total objective summed over all target items."""
    X = to_dense(X)
    W = to_dense(W)
    residual = X - X @ W
    return float(
        0.5 * np.sum(residual**2) + lambd * np.sum(W) + 0.5 * beta * np.sum(W**2)
    )


def kkt_violation(X, W, lambd: float, beta: float) -> float:
    """Largest KKT violation of ``W`` for the SLIM objective.

    For target item ``i`` and neighbour ``k != i``, with
    ``g_ik = P_ik - sum_j w_ji P_jk``:

    * an active weight (``w_ki > 0``) must satisfy ``g_ik - lambd - beta * w_ki == 0``;
    * an inactive weight (``w_ki == 0``) must satisfy ``g_ik - lambd <= 0``.

    The return value is the largest absolute breach of those conditions, so an
    exact solution scores ``0``.
    """
    X = to_dense(X)
    W = to_dense(W)
    if W.size == 0:
        return 0.0
    P = X.T @ X
    # gradient[k, i] = P_ik - sum_j w_ji P_jk, using the symmetry of P.
    gradient = P - P @ W
    violation = np.where(
        W > 0,
        np.abs(gradient - lambd - beta * W),
        np.maximum(0.0, gradient - lambd),
    )
    np.fill_diagonal(violation, 0.0)
    return float(violation.max())


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator, so every test draws the same data on every run."""
    return np.random.default_rng(20240501)


@pytest.fixture
def make_binary():
    """Factory for random binary user-item CSR matrices."""

    def make(
        n_users: int = 80,
        n_items: int = 30,
        density: float = 0.15,
        seed: int = 0,
    ) -> sparse.csr_matrix:
        generator = np.random.default_rng(seed)
        dense = (generator.random((n_users, n_items)) < density).astype(np.float64)
        return sparse.csr_matrix(dense)

    return make


@pytest.fixture
def make_weighted():
    """Factory for random non-negative float user-item CSR matrices."""

    def make(
        n_users: int = 80,
        n_items: int = 30,
        density: float = 0.15,
        seed: int = 0,
        high: float = 5.0,
    ) -> sparse.csr_matrix:
        generator = np.random.default_rng(seed)
        mask = generator.random((n_users, n_items)) < density
        values = generator.uniform(0.5, high, size=mask.shape)
        return sparse.csr_matrix(np.where(mask, values, 0.0))

    return make


@pytest.fixture
def binary_matrix(make_binary) -> sparse.csr_matrix:
    """A small binary interaction matrix used by most tests."""
    return make_binary(n_users=60, n_items=20, density=0.2, seed=1)


@pytest.fixture
def tiny_matrix() -> sparse.csr_matrix:
    """Six users over four items, dense enough that every item has neighbours."""
    dense = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 1],
        ],
        dtype=np.float64,
    )
    return sparse.csr_matrix(dense)
