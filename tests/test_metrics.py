"""Ranking metrics, checked against arithmetic done by hand."""

from __future__ import annotations

import fastslim
import numpy as np
import pytest
from fastslim import metrics
from scipy import sparse

N_ITEMS = 5

# Three users; user 1 has nothing held out and must be skipped everywhere.
TEST_ROWS = {0: [1, 3], 1: [], 2: [0, 2, 4]}

# Ranked recommendations, best first.
RECOMMENDED = np.array(
    [
        [3, 0, 1],  # user 0: hits at ranks 0 and 2
        [0, 1, 2],  # user 1: skipped
        [1, 4, 0],  # user 2: hits at ranks 1 and 2
    ]
)

K = 3
DISCOUNT = 1.0 / np.log2(np.arange(K) + 2.0)  # [1, 0.63093, 0.5]


@pytest.fixture
def test_matrix() -> sparse.csr_matrix:
    dense = np.zeros((3, N_ITEMS))
    for user, items in TEST_ROWS.items():
        dense[user, items] = 1.0
    return sparse.csr_matrix(dense)


@pytest.fixture
def scores() -> np.ndarray:
    """Scores whose per-user ranking is exactly ``RECOMMENDED``."""
    dense = np.full((3, N_ITEMS), -100.0)
    for user, ranked in enumerate(RECOMMENDED):
        dense[user, ranked] = [3.0, 2.0, 1.0]
    return dense


def test_precision_at_k(test_matrix):
    # user 0: 2 of 3 recommendations are held out; user 2: 2 of 3.  Both
    # denominators are k, not the number of held-out items.
    expected = np.mean([2 / 3, 2 / 3])
    assert metrics.precision_at_k(RECOMMENDED, test_matrix, k=K) == pytest.approx(
        expected
    )


def test_recall_at_k(test_matrix):
    # user 0 found 2 of its 2 held-out items; user 2 found 2 of its 3.
    expected = np.mean([2 / 2, 2 / 3])
    assert metrics.recall_at_k(RECOMMENDED, test_matrix, k=K) == pytest.approx(expected)


def test_ndcg_at_k(test_matrix):
    # user 0 hits ranks 0 and 2 against an ideal of ranks 0 and 1.
    ndcg_0 = (DISCOUNT[0] + DISCOUNT[2]) / (DISCOUNT[0] + DISCOUNT[1])
    # user 2 hits ranks 1 and 2 against an ideal of ranks 0, 1 and 2.
    ndcg_2 = (DISCOUNT[1] + DISCOUNT[2]) / DISCOUNT.sum()
    expected = np.mean([ndcg_0, ndcg_2])
    assert metrics.ndcg_at_k(RECOMMENDED, test_matrix, k=K) == pytest.approx(expected)


@pytest.mark.parametrize(
    "metric", [metrics.precision_at_k, metrics.recall_at_k, metrics.ndcg_at_k]
)
def test_scores_and_recommendations_agree(metric, scores, test_matrix):
    assert metric(scores, test_matrix, k=K) == pytest.approx(
        metric(RECOMMENDED, test_matrix, k=K)
    )


@pytest.mark.parametrize(
    "metric", [metrics.precision_at_k, metrics.recall_at_k, metrics.ndcg_at_k]
)
def test_users_without_test_items_are_skipped(metric, test_matrix):
    """Adding users with an empty test row must not move the numbers."""
    padded_test = sparse.vstack(
        [test_matrix, sparse.csr_matrix((4, N_ITEMS))], format="csr"
    )
    padded_recs = np.vstack([RECOMMENDED, np.zeros((4, K), dtype=int)])

    assert metric(padded_recs, padded_test, k=K) == pytest.approx(
        metric(RECOMMENDED, test_matrix, k=K)
    )


@pytest.mark.parametrize(
    "metric", [metrics.precision_at_k, metrics.recall_at_k, metrics.ndcg_at_k]
)
def test_no_evaluable_users_gives_zero(metric):
    empty_test = sparse.csr_matrix((3, N_ITEMS))
    assert metric(RECOMMENDED, empty_test, k=K) == 0.0


def test_perfect_ranking(test_matrix):
    """Held-out items first: recall and NDCG reach 1, precision cannot."""
    perfect = np.array([[1, 3, 4], [0, 1, 2], [0, 2, 4]])

    assert metrics.recall_at_k(perfect, test_matrix, k=K) == pytest.approx(1.0)
    assert metrics.ndcg_at_k(perfect, test_matrix, k=K) == pytest.approx(1.0)
    # User 0 only has 2 held-out items, so 3 slots can never all be relevant.
    assert metrics.precision_at_k(perfect, test_matrix, k=K) == pytest.approx(
        np.mean([2 / 3, 3 / 3])
    )


@pytest.mark.parametrize(
    "metric", [metrics.precision_at_k, metrics.recall_at_k, metrics.ndcg_at_k]
)
def test_ranking_with_no_hits_scores_zero(metric, test_matrix):
    # User 2 has only two non-held-out items, so its third slot repeats one;
    # duplicates are irrelevant here because neither is a hit.
    nothing = np.array([[0, 2, 4], [0, 1, 2], [1, 3, 3]])
    assert metric(nothing, test_matrix, k=K) == 0.0


def test_ndcg_rewards_putting_hits_first(test_matrix):
    early = np.array([[1, 3, 0], [0, 0, 0], [0, 2, 4]])
    late = np.array([[0, 1, 3], [0, 0, 0], [1, 0, 2]])
    assert metrics.ndcg_at_k(early, test_matrix, k=K) > metrics.ndcg_at_k(
        late, test_matrix, k=K
    )


def test_k_larger_than_the_score_matrix_is_clipped(scores, test_matrix):
    """Only ``N_ITEMS`` items exist, so k=10 can rank at most that many."""
    assert metrics.recall_at_k(scores, test_matrix, k=10) == pytest.approx(1.0)


def test_recommendation_array_must_be_long_enough(test_matrix):
    with pytest.raises(ValueError, match="need at least k="):
        metrics.precision_at_k(RECOMMENDED, test_matrix, k=4)


def test_shape_mismatch_is_reported(test_matrix):
    with pytest.raises(ValueError, match="but predictions cover"):
        metrics.recall_at_k(RECOMMENDED[:2], test_matrix, k=K)


def test_metrics_accept_fastslim_output(tiny_matrix):
    """End-to-end: fit, score, rank, measure -- shapes and ranges line up."""
    W = fastslim.fit(tiny_matrix, lambd=0.2, beta=0.2)
    # Score against the training interactions themselves, so there is something
    # to hit; masking them would make every metric trivially zero.
    scores = fastslim.predict(W, tiny_matrix, exclude_seen=False)
    top = fastslim.recommend(W, tiny_matrix, k=2, exclude_seen=False)

    for metric in (metrics.precision_at_k, metrics.recall_at_k, metrics.ndcg_at_k):
        from_scores = metric(scores, tiny_matrix, k=2)
        from_recs = metric(top, tiny_matrix, k=2)
        assert 0.0 < from_scores <= 1.0
        assert from_scores == pytest.approx(from_recs)
