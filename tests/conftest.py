from __future__ import annotations

from typing import Any

import fastslim
import numpy as np
import pytest
from scipy import sparse

TINY = np.array(
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


def to_dense(matrix: Any) -> np.ndarray:
    """Dense float64 view of a sparse or dense matrix."""
    if sparse.issparse(matrix):
        return np.asarray(matrix.todense(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def random_matrix(
    n_users: int = 80,
    n_items: int = 30,
    density: float = 0.15,
    seed: int = 0,
    high: float = 0.0,
) -> sparse.csr_matrix:
    """Random non-negative CSR matrix: 0/1 values, or floats in ``[0.5, high)``."""
    generator = np.random.default_rng(seed)
    mask = generator.random((n_users, n_items)) < density
    values = generator.uniform(0.5, high, size=mask.shape) if high else 1.0
    return sparse.csr_matrix(np.where(mask, values, 0.0))


def reference_cd(
    dense: np.ndarray,
    lambd: float,
    beta: float,
    max_passes: int = 5000,
    tol: float = 1e-13,
) -> np.ndarray:
    """Solve the SLIM problem by naive full-pass non-negative coordinate descent.

    Returns ``W[k, i] = w_ik``, exactly as :func:`fastslim.fit` does.
    """
    dense = np.asarray(dense, dtype=np.float64)
    gram = dense.T @ dense
    n_items = gram.shape[0]
    weights = np.zeros((n_items, n_items), dtype=np.float64)
    for i in range(n_items):
        w = np.zeros(n_items)
        for _ in range(max_passes):
            max_delta = 0.0
            for k in range(n_items):
                denominator = gram[k, k] + beta
                if k == i or denominator == 0.0:
                    continue
                # Partial residual correlation with coordinate k held out.
                correlation = gram[i, k] - (gram[k] @ w) + gram[k, k] * w[k]
                new = max(0.0, (correlation - lambd) / denominator)
                delta = new - w[k]
                if delta != 0.0:
                    w[k] = new
                    max_delta = max(max_delta, abs(delta))
            if max_delta < tol:
                break
        weights[:, i] = w
    return weights


def objective(interactions: Any, weights: Any, lambd: float, beta: float) -> float:
    """Total SLIM objective summed over all target items."""
    interactions = to_dense(interactions)
    weights = to_dense(weights)
    residual = interactions - interactions @ weights
    return float(
        0.5 * np.sum(residual**2)
        + lambd * np.sum(weights)
        + 0.5 * beta * np.sum(weights**2)
    )


def kkt_violation(interactions: Any, weights: Any, lambd: float, beta: float) -> float:
    """Largest KKT violation of ``weights``; an exact solution scores ``0``.

    With ``g_ik = P_ik - sum_j w_ji P_jk``, an active weight must satisfy
    ``g_ik - lambd - beta * w_ki == 0``, an inactive one ``g_ik - lambd <= 0``.
    """
    interactions = to_dense(interactions)
    weights = to_dense(weights)
    if weights.size == 0:
        return 0.0
    gram = interactions.T @ interactions
    gradient = gram - gram @ weights
    violation = np.where(
        weights > 0,
        np.abs(gradient - lambd - beta * weights),
        np.maximum(0.0, gradient - lambd),
    )
    np.fill_diagonal(violation, 0.0)
    return float(violation.max())


@pytest.fixture
def binary_matrix() -> sparse.csr_matrix:
    """Build the small binary interaction matrix used by most tests."""
    return random_matrix(n_users=60, n_items=20, density=0.2, seed=1)


@pytest.fixture
def tiny_matrix() -> sparse.csr_matrix:
    """Six users over four items, dense enough that every item has neighbours."""
    return sparse.csr_matrix(TINY)


@pytest.fixture
def fitted(tiny_matrix) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """``(interactions, weights)`` for a converged fit on ``tiny_matrix``."""
    return tiny_matrix, fastslim.fit(tiny_matrix, lambd=0.2, beta=0.2)


@pytest.fixture
def ill_conditioned() -> sparse.csr_matrix:
    """Two identical popular columns, which makes coordinate descent crawl.

    Each pass shrinks item 2's error only by ``(P / (P + beta)) ** 2``, so it
    needs hundreds of passes at the default ``tol``.
    """
    dense = np.zeros((60, 4))
    dense[:50, 0] = 1.0
    dense[:50, 1] = 1.0
    dense[:40, 2] = 1.0
    dense[10:30, 3] = 1.0
    return sparse.csr_matrix(dense)
