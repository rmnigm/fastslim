"""Scoring and ranking: ``predict`` and ``recommend``."""

from __future__ import annotations

import fastslim
import numpy as np
import pytest
from conftest import to_dense
from scipy import sparse


@pytest.fixture
def model(tiny_matrix):
    return tiny_matrix, fastslim.fit(tiny_matrix, lambd=0.2, beta=0.2)


def test_scores_are_exactly_history_times_weights(model):
    X, W = model
    expected = to_dense(X) @ to_dense(W)

    got = fastslim.predict(W, X, exclude_seen=False)

    np.testing.assert_allclose(got, expected)
    assert got.dtype == np.float64


def test_sparse_and_dense_history_agree(model):
    X, W = model
    from_sparse = fastslim.predict(W, X)
    from_dense = fastslim.predict(W, to_dense(X))
    np.testing.assert_allclose(from_sparse, from_dense)


@pytest.mark.parametrize("exclude_seen", [True, False])
def test_single_history_matches_the_batch_row(model, exclude_seen):
    X, W = model
    batch = fastslim.predict(W, X, exclude_seen=exclude_seen)

    for user in range(X.shape[0]):
        as_sparse_row = fastslim.predict(W, X[user], exclude_seen=exclude_seen)
        as_dense_row = fastslim.predict(W, to_dense(X)[user], exclude_seen=exclude_seen)
        assert as_sparse_row.shape == (X.shape[1],)
        assert as_dense_row.shape == (X.shape[1],)
        np.testing.assert_array_equal(as_sparse_row, batch[user])
        np.testing.assert_array_equal(as_dense_row, batch[user])


def test_exclude_seen_masks_exactly_the_nonzero_entries(model):
    X, W = model
    scores = fastslim.predict(W, X, exclude_seen=True)

    masked = np.isneginf(scores)
    np.testing.assert_array_equal(masked, to_dense(X) != 0)
    # The unmasked half is untouched.
    unmasked = fastslim.predict(W, X, exclude_seen=False)
    np.testing.assert_allclose(scores[~masked], unmasked[~masked])


def test_a_sparse_array_row_stays_a_batch(model):
    """``csr_array`` can be 1-D, so ``(1, n)`` means one-user batch, not a vector."""
    X, W = model
    X_array = sparse.csr_array(X)
    assert fastslim.predict(W, X_array[[0]]).shape == (1, X.shape[1])
    assert fastslim.predict(W, X_array[0]).shape == (X.shape[1],)


@pytest.mark.parametrize("batch_size", [1, 2, 5, 100])
def test_batching_does_not_change_results(model, batch_size):
    X, W = model
    np.testing.assert_array_equal(
        fastslim.predict(W, X, batch_size=batch_size),
        fastslim.predict(W, X),
    )
    np.testing.assert_array_equal(
        fastslim.recommend(W, X, k=3, batch_size=batch_size),
        fastslim.recommend(W, X, k=3),
    )


def test_recommend_returns_top_k_in_descending_order(model):
    X, W = model
    k = 3
    scores = fastslim.predict(W, X)
    top = fastslim.recommend(W, X, k=k)

    assert top.shape == (X.shape[0], k)
    assert top.dtype == np.int64
    for user in range(X.shape[0]):
        chosen = scores[user, top[user]]
        # Compared pairwise rather than with np.diff, which turns -inf minus
        # -inf into nan when a user has seen more items than k leaves unseen.
        assert np.all(chosen[:-1] >= chosen[1:]), "scores must be non-increasing"
        # No item outside the selection beats the worst selected one.
        rest = np.setdiff1d(np.arange(X.shape[1]), top[user])
        if rest.size:
            assert scores[user, rest].max() <= chosen.min()


def test_recommend_never_returns_seen_items(model):
    X, W = model
    # Every user has seen at most 3 of the 4 items, so k=1 is always safe.
    top = fastslim.recommend(W, X, k=1)
    for user in range(X.shape[0]):
        assert X[user, top[user, 0]] == 0


def test_recommend_single_history_shape(model):
    X, W = model
    assert fastslim.recommend(W, X[0], k=2).shape == (2,)
    assert fastslim.recommend(W, to_dense(X)[0], k=2).shape == (2,)
    np.testing.assert_array_equal(
        fastslim.recommend(W, X[0], k=2), fastslim.recommend(W, X, k=2)[0]
    )


def test_k_larger_than_n_items_is_clipped(model):
    X, W = model
    top = fastslim.recommend(W, X, k=1000, exclude_seen=False)
    assert top.shape == (X.shape[0], X.shape[1])
    for user in range(X.shape[0]):
        assert sorted(top[user]) == list(range(X.shape[1]))


def test_ties_are_broken_by_ascending_item_index():
    """With an all-zero weight matrix every score ties; order must be stable."""
    W = sparse.csr_matrix((5, 5))
    history = np.zeros(5)
    np.testing.assert_array_equal(fastslim.recommend(W, history, k=5), np.arange(5))


def test_recommend_falls_back_to_seen_items_when_it_must(model):
    """Asking for more items than are unseen fills the tail with seen ones."""
    X, W = model
    user = 1  # has seen 3 of 4 items
    top = fastslim.recommend(W, X[user], k=4)
    assert sorted(top) == [0, 1, 2, 3]
    assert X[user, top[0]] == 0, "the one unseen item still ranks first"


def test_dense_weights_are_accepted(model):
    X, W = model
    np.testing.assert_allclose(fastslim.predict(to_dense(W), X), fastslim.predict(W, X))


def test_zero_users_batch(model):
    X, W = model
    empty = sparse.csr_matrix((0, X.shape[1]))
    assert fastslim.predict(W, empty).shape == (0, X.shape[1])
    assert fastslim.recommend(W, empty, k=2).shape == (0, 2)
