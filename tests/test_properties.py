from __future__ import annotations

import fastslim
import numpy as np
from conftest import kkt_violation, to_dense
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst
from scipy import sparse

# deadline=None: a first call pays import and thread-pool costs.
SETTINGS = settings(max_examples=30, deadline=None)

# Small, non-negative, finite matrices; values coarse enough that ties and
# exact zeros show up often.
matrices = npst.arrays(
    dtype=np.float64,
    shape=st.tuples(st.integers(0, 8), st.integers(0, 6)),
    elements=st.sampled_from([0.0, 0.0, 0.0, 1.0, 2.0, 0.5]),
)

penalties = st.floats(min_value=0.0, max_value=3.0, allow_nan=False)


@SETTINGS
@given(dense=matrices, lambd=penalties, beta=penalties)
def test_structural_invariants(dense, lambd, beta):
    weights = fastslim.fit(
        sparse.csr_matrix(dense), lambd=lambd, beta=beta, max_iter=5000
    )

    assert weights.shape == (dense.shape[1], dense.shape[1])
    assert np.all(weights.data > 0), "only strictly positive weights are stored"
    assert np.count_nonzero(weights.diagonal()) == 0, "an item never predicts itself"
    assert weights.has_canonical_format


@SETTINGS
@given(dense=matrices, lambd=penalties, beta=penalties)
def test_solution_is_optimal(dense, lambd, beta):
    weights = fastslim.fit(
        sparse.csr_matrix(dense), lambd=lambd, beta=beta, tol=1e-12, max_iter=5000
    )
    assert kkt_violation(dense, weights, lambd, beta) < 1e-6


@SETTINGS
@given(dense=matrices)
def test_scores_and_ranking_agree(dense):
    weights = fastslim.fit(sparse.csr_matrix(dense), lambd=0.3, beta=0.3, max_iter=5000)
    n_items = dense.shape[1]
    if n_items == 0 or dense.shape[0] == 0:
        return

    scores = fastslim.predict(weights, dense)
    top = fastslim.recommend(weights, dense, k=n_items)

    np.testing.assert_allclose(
        fastslim.predict(weights, dense, exclude_seen=False), dense @ to_dense(weights)
    )
    for user in range(dense.shape[0]):
        ranked = scores[user, top[user]]
        assert np.all(ranked[:-1] >= ranked[1:])
        assert sorted(top[user]) == list(range(n_items))


@SETTINGS
@given(dense=matrices)
def test_column_permutation_permutes_the_weights(dense):
    """Item identity is arbitrary: relabel the items, relabel the model.

    The two runs visit coordinates in different orders and so stop at slightly
    different points, hence a tolerance above machine precision.
    """
    n_items = dense.shape[1]
    if n_items == 0:
        return
    permutation = np.arange(n_items)[::-1]
    params = {"lambd": 0.4, "beta": 0.4, "tol": 1e-12, "max_iter": 5000}

    direct = to_dense(fastslim.fit(sparse.csr_matrix(dense), **params))
    permuted = to_dense(
        fastslim.fit(sparse.csr_matrix(dense[:, permutation]), **params)
    )

    np.testing.assert_allclose(
        permuted, direct[np.ix_(permutation, permutation)], atol=1e-8
    )
