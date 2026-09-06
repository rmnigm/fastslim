from __future__ import annotations

import fastslim
import numpy as np
import pytest
from conftest import kkt_violation, objective, random_matrix, reference_cd, to_dense
from scipy import sparse

# With tol=1e-10 the observed worst-case KKT violation across this grid is
# ~2e-9, so 1e-6 leaves a wide margin without being vacuous.
EXACT = {"tol": 1e-10, "max_iter": 10_000}
KKT_TOL = 1e-6

LAMBDS = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0]
BETAS = [0.0, 0.5, 2.0]
PENALTIES = [(0.0, 0.5), (0.1, 0.0), (0.5, 0.5), (1.0, 2.0), (2.0, 0.5), (3.0, 0.0)]


@pytest.fixture(params=["binary", "weighted"])
def interactions(request) -> sparse.csr_matrix:
    """Build a binary and a float-valued interaction matrix of the same shape."""
    high = 5.0 if request.param == "weighted" else 0.0
    seed = 2 if request.param == "weighted" else 1
    return random_matrix(n_users=60, n_items=20, density=0.2, seed=seed, high=high)


@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("lambd", LAMBDS)
def test_solution_satisfies_kkt_conditions(interactions, lambd, beta):
    weights = fastslim.fit(interactions, lambd=lambd, beta=beta, **EXACT)
    assert kkt_violation(interactions, weights, lambd, beta) < KKT_TOL


@pytest.mark.parametrize(("lambd", "beta"), PENALTIES)
def test_matches_reference_solver(lambd, beta):
    dense = to_dense(random_matrix(n_users=40, n_items=12, density=0.25, seed=3))
    got = to_dense(fastslim.fit(dense, lambd=lambd, beta=beta, **EXACT))
    expected = reference_cd(dense, lambd, beta)

    assert np.array_equal(got > 0, expected > 0)
    assert np.allclose(got, expected, atol=1e-8, rtol=0)


@pytest.mark.parametrize(
    ("lambd", "beta"), [(0.0, 0.5), (0.5, 0.5), (1.0, 0.0), (2.0, 2.0)]
)
def test_objective_no_worse_than_reference(lambd, beta):
    dense = to_dense(
        random_matrix(n_users=40, n_items=12, density=0.25, seed=4, high=5.0)
    )
    got = fastslim.fit(dense, lambd=lambd, beta=beta, **EXACT)
    expected = reference_cd(dense, lambd, beta)

    assert objective(dense, got, lambd, beta) <= (
        objective(dense, expected, lambd, beta) + 1e-9
    )


def test_hand_computed_three_items():
    """Three items whose optimum can be written down by hand.

    Columns (1,1,0), (1,1,1), (0,1,1) give ``P = [[2,2,1],[2,3,2],[1,2,2]]``, so
    stationarity puts every active weight at ``t = 1.9 / 3.1``.
    """
    interactions = sparse.csr_matrix(
        np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    )
    t = 1.9 / 3.1
    expected = np.array([[0.0, t, 0.0], [t, 0.0, t], [0.0, t, 0.0]])

    weights = fastslim.fit(interactions, lambd=0.1, beta=0.1, **EXACT)
    np.testing.assert_allclose(to_dense(weights), expected, atol=1e-10)


def test_stored_weights_are_positive_and_off_diagonal(binary_matrix):
    weights = fastslim.fit(binary_matrix, lambd=0.2, beta=0.2)
    assert weights.nnz > 0
    assert np.all(weights.data > 0)
    assert np.count_nonzero(weights.diagonal()) == 0


def test_huge_lambd_prunes_everything(binary_matrix):
    weights = fastslim.fit(binary_matrix, lambd=1e6, beta=0.5)
    assert weights.nnz == 0
    assert weights.shape == (binary_matrix.shape[1],) * 2


def test_larger_lambd_never_adds_weights(binary_matrix):
    """Sparsity is monotone in the L1 penalty on this fixed matrix."""
    counts = [
        fastslim.fit(binary_matrix, lambd=lambd, beta=0.5, **EXACT).nnz
        for lambd in LAMBDS
    ]
    assert counts == sorted(counts, reverse=True)
