"""Does ``fit`` actually solve the optimisation problem it documents?

Every assertion here is against an independent description of the problem --
the KKT conditions, a naive reference solver, or arithmetic done by hand -- so
that a change in the Rust solver that alters the answer is caught even if it is
internally self-consistent.
"""

from __future__ import annotations

import fastslim
import numpy as np
import pytest
from conftest import kkt_violation, objective, reference_cd, to_dense
from scipy import sparse

# Tolerances: with tol=1e-10 the observed worst-case KKT violation across this
# grid is ~2e-9, so 1e-6 leaves a wide margin without being vacuous.
EXACT = {"tol": 1e-10, "max_iter": 10_000}
KKT_TOL = 1e-6

LAMBDS = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0]
BETAS = [0.0, 0.5, 2.0]


@pytest.fixture
def matrices(make_binary, make_weighted):
    return {
        "binary": make_binary(n_users=60, n_items=20, density=0.2, seed=1),
        "weighted": make_weighted(n_users=60, n_items=20, density=0.2, seed=2),
    }


@pytest.mark.parametrize("kind", ["binary", "weighted"])
@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("lambd", LAMBDS)
def test_solution_satisfies_kkt_conditions(matrices, kind, lambd, beta):
    X = matrices[kind]
    W = fastslim.fit(X, lambd=lambd, beta=beta, **EXACT)
    assert kkt_violation(X, W, lambd, beta) < KKT_TOL


@pytest.mark.parametrize(
    ("lambd", "beta"),
    [(0.0, 0.5), (0.1, 0.0), (0.5, 0.5), (1.0, 2.0), (2.0, 0.5), (3.0, 0.0)],
)
def test_matches_reference_solver(make_binary, lambd, beta):
    X = make_binary(n_users=40, n_items=12, density=0.25, seed=3)
    dense = to_dense(X)
    got = to_dense(fastslim.fit(X, lambd=lambd, beta=beta, **EXACT))
    expected = reference_cd(dense, lambd, beta)

    assert np.count_nonzero(got) == np.count_nonzero(expected)
    assert np.array_equal(got > 0, expected > 0)
    assert np.allclose(got, expected, atol=1e-8, rtol=0)


@pytest.mark.parametrize(
    ("lambd", "beta"),
    [(0.0, 0.5), (0.5, 0.5), (1.0, 0.0), (2.0, 2.0)],
)
def test_objective_no_worse_than_reference(make_weighted, lambd, beta):
    X = make_weighted(n_users=40, n_items=12, density=0.25, seed=4)
    dense = to_dense(X)
    got = fastslim.fit(X, lambd=lambd, beta=beta, **EXACT)
    expected = reference_cd(dense, lambd, beta)

    assert objective(dense, got, lambd, beta) <= (
        objective(dense, expected, lambd, beta) + 1e-9
    )


def test_hand_computed_three_items():
    """Three items where the optimum can be written down.

    ``X`` has columns ``(1,1,0)``, ``(1,1,1)``, ``(0,1,1)``, so the Gram matrix
    is ``P = [[2,2,1],[2,3,2],[1,2,2]]``.  For target item 0 the candidate
    ``w = (0, t, 0)`` satisfies stationarity when
    ``P01 - t*P11 - lambd - beta*t = 0``, i.e. ``t = (2 - 0.1) / (3 + 0.1)``,
    and item 2 stays inactive because
    ``P02 - t*P12 - lambd = 1 - 2t - 0.1 < 0``.  Item 2 mirrors item 0.  For
    target item 1 symmetry gives ``w0 = w2 = s`` with
    ``P01 - s*P00 - s*P02 - lambd - beta*s = 0``, i.e.
    ``s = (2 - 0.1) / (2 + 1 + 0.1)``, which is the same number.
    """
    X = sparse.csr_matrix(np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]))
    t = 1.9 / 3.1
    expected = np.array([[0.0, t, 0.0], [t, 0.0, t], [0.0, t, 0.0]])

    W = fastslim.fit(X, lambd=0.1, beta=0.1, **EXACT)
    np.testing.assert_allclose(to_dense(W), expected, atol=1e-10)


def test_diagonal_is_zero(binary_matrix):
    W = fastslim.fit(binary_matrix, lambd=0.2, beta=0.2)
    assert np.count_nonzero(W.diagonal()) == 0


def test_stored_weights_are_strictly_positive(binary_matrix):
    W = fastslim.fit(binary_matrix, lambd=0.2, beta=0.2)
    assert W.nnz > 0
    assert np.all(W.data > 0)


def test_huge_lambd_prunes_everything(binary_matrix):
    W = fastslim.fit(binary_matrix, lambd=1e6, beta=0.5)
    assert W.nnz == 0
    assert W.shape == (binary_matrix.shape[1],) * 2


def test_max_iter_zero_gives_empty_weights(binary_matrix):
    W = fastslim.fit(binary_matrix, lambd=0.1, beta=0.1, max_iter=0)
    assert W.nnz == 0


def test_larger_lambd_never_adds_weights(binary_matrix):
    """Sparsity is monotone in the L1 penalty on this (fixed) matrix.

    Monotone support is not guaranteed in general for a non-negative
    elastic net, but it holds here and a change that broke it would mean
    the solver had started returning something quite different.
    """
    counts = [
        fastslim.fit(binary_matrix, lambd=lambd, beta=0.5, **EXACT).nnz
        for lambd in LAMBDS
    ]
    assert counts == sorted(counts, reverse=True)
