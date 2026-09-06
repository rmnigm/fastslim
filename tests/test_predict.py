from __future__ import annotations

import fastslim
import numpy as np
import pytest
from conftest import to_dense
from scipy import sparse


def test_scores_are_exactly_history_times_weights(fitted):
    history, weights = fitted
    got = fastslim.predict(weights, history, exclude_seen=False)
    np.testing.assert_allclose(got, to_dense(history) @ to_dense(weights))
    assert got.dtype == np.float64


def test_sparse_and_dense_history_agree(fitted):
    history, weights = fitted
    np.testing.assert_allclose(
        fastslim.predict(weights, history),
        fastslim.predict(weights, to_dense(history)),
    )


def test_dense_weights_are_accepted(fitted):
    history, weights = fitted
    np.testing.assert_allclose(
        fastslim.predict(to_dense(weights), history),
        fastslim.predict(weights, history),
    )


@pytest.mark.parametrize("exclude_seen", [True, False], ids=["masked", "unmasked"])
@pytest.mark.parametrize("as_dense", [True, False], ids=["dense_row", "sparse_row"])
def test_single_history_matches_the_batch_row(fitted, exclude_seen, as_dense):
    history, weights = fitted
    batch = fastslim.predict(weights, history, exclude_seen=exclude_seen)
    rows = to_dense(history) if as_dense else history

    for user in range(history.shape[0]):
        row = fastslim.predict(weights, rows[user], exclude_seen=exclude_seen)
        assert row.shape == (history.shape[1],)
        np.testing.assert_array_equal(row, batch[user])


def test_exclude_seen_masks_exactly_the_nonzero_entries(fitted):
    history, weights = fitted
    scores = fastslim.predict(weights, history, exclude_seen=True)

    masked = np.isneginf(scores)
    np.testing.assert_array_equal(masked, to_dense(history) != 0)
    # The unmasked half is untouched.
    unmasked = fastslim.predict(weights, history, exclude_seen=False)
    np.testing.assert_allclose(scores[~masked], unmasked[~masked])


def test_a_sparse_array_row_stays_a_batch(fitted):
    """``csr_array`` can be 1-D, so ``(1, n)`` means one-user batch, not a vector."""
    history, weights = fitted
    as_array = sparse.csr_array(history)
    assert fastslim.predict(weights, as_array[[0]]).shape == (1, history.shape[1])
    assert fastslim.predict(weights, as_array[0]).shape == (history.shape[1],)


@pytest.mark.parametrize("batch_size", [1, 2, 5, 100])
@pytest.mark.parametrize("call", ["predict", "recommend"])
def test_batching_does_not_change_results(fitted, batch_size, call):
    history, weights = fitted
    kwargs = {"k": 3} if call == "recommend" else {}
    np.testing.assert_array_equal(
        getattr(fastslim, call)(weights, history, batch_size=batch_size, **kwargs),
        getattr(fastslim, call)(weights, history, **kwargs),
    )


def test_recommend_returns_top_k_in_descending_order(fitted):
    history, weights = fitted
    k = 3
    scores = fastslim.predict(weights, history)
    top = fastslim.recommend(weights, history, k=k)

    assert top.shape == (history.shape[0], k)
    assert top.dtype == np.int64
    for user in range(history.shape[0]):
        chosen = scores[user, top[user]]
        # Compared pairwise rather than with np.diff, which turns -inf minus
        # -inf into nan when a user has seen more items than k leaves unseen.
        assert np.all(chosen[:-1] >= chosen[1:]), "scores must be non-increasing"
        rest = np.setdiff1d(np.arange(history.shape[1]), top[user])
        if rest.size:
            assert scores[user, rest].max() <= chosen.min()


def test_recommend_never_returns_seen_items(fitted):
    history, weights = fitted
    # Every user has seen at most 3 of the 4 items, so k=1 is always safe.
    top = fastslim.recommend(weights, history, k=1)
    for user in range(history.shape[0]):
        assert history[user, top[user, 0]] == 0


@pytest.mark.parametrize("as_dense", [True, False], ids=["dense", "sparse"])
def test_recommend_single_history_shape(fitted, as_dense):
    history, weights = fitted
    row = to_dense(history)[0] if as_dense else history[0]
    top = fastslim.recommend(weights, row, k=2)
    assert top.shape == (2,)
    np.testing.assert_array_equal(top, fastslim.recommend(weights, history, k=2)[0])


def test_k_larger_than_n_items_is_clipped(fitted):
    history, weights = fitted
    top = fastslim.recommend(weights, history, k=1000, exclude_seen=False)
    assert top.shape == history.shape
    for user in range(history.shape[0]):
        assert sorted(top[user]) == list(range(history.shape[1]))


def test_ties_are_broken_by_ascending_item_index():
    """With an all-zero weight matrix every score ties; order must be stable."""
    weights = sparse.csr_matrix((5, 5))
    np.testing.assert_array_equal(
        fastslim.recommend(weights, np.zeros(5), k=5), np.arange(5)
    )


def test_recommend_falls_back_to_seen_items_when_it_must(fitted):
    """Asking for more items than are unseen fills the tail with seen ones."""
    history, weights = fitted
    user = 1  # has seen 3 of 4 items
    top = fastslim.recommend(weights, history[user], k=4)
    assert sorted(top) == [0, 1, 2, 3]
    assert history[user, top[0]] == 0, "the one unseen item still ranks first"


@pytest.mark.parametrize("call", ["predict", "recommend"])
def test_zero_users_batch(fitted, call):
    history, weights = fitted
    empty = sparse.csr_matrix((0, history.shape[1]))
    kwargs = {"k": 2} if call == "recommend" else {}
    expected = (0, 2) if call == "recommend" else (0, history.shape[1])
    assert getattr(fastslim, call)(weights, empty, **kwargs).shape == expected
