from __future__ import annotations

import fastslim
import numpy as np
import pytest
from fastslim import metrics
from scipy import sparse

N_ITEMS = 5
K = 3
METRICS = [metrics.precision_at_k, metrics.recall_at_k, metrics.ndcg_at_k]
METRIC_IDS = ["precision", "recall", "ndcg"]

# Three users; user 1 has nothing held out and must be skipped everywhere.
TEST_ROWS = {0: [1, 3], 1: [], 2: [0, 2, 4]}

# Ranked recommendations, best first: user 0 hits at ranks 0 and 2, user 2 at
# ranks 1 and 2.
RECOMMENDED = np.array([[3, 0, 1], [0, 1, 2], [1, 4, 0]])

DISCOUNT = 1.0 / np.log2(np.arange(K) + 2.0)  # [1, 0.63093, 0.5]


@pytest.fixture
def test_matrix() -> sparse.csr_matrix:
    """Held-out interactions matching ``TEST_ROWS``."""
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


# Precision divides by k, not by the number of held-out items.  NDCG weighs
# user 0's hits at ranks 0 and 2 against an ideal of ranks 0 and 1, and user
# 2's hits at ranks 1 and 2 against an ideal of ranks 0, 1 and 2.
HAND_COMPUTED = [
    np.mean([2 / 3, 2 / 3]),
    np.mean([2 / 2, 2 / 3]),
    np.mean(
        [
            (DISCOUNT[0] + DISCOUNT[2]) / (DISCOUNT[0] + DISCOUNT[1]),
            (DISCOUNT[1] + DISCOUNT[2]) / DISCOUNT.sum(),
        ]
    ),
]

# User 0 has only 2 held-out items, so 3 slots can never all be relevant.
PERFECT = [np.mean([2 / 3, 3 / 3]), 1.0, 1.0]


@pytest.mark.parametrize(
    ("metric", "expected"),
    list(zip(METRICS, HAND_COMPUTED, strict=True)),
    ids=METRIC_IDS,
)
def test_metric_matches_hand_computed_value(metric, expected, test_matrix):
    assert metric(RECOMMENDED, test_matrix, k=K) == pytest.approx(expected)


@pytest.mark.parametrize("metric", METRICS, ids=METRIC_IDS)
def test_scores_and_recommendations_agree(metric, scores, test_matrix):
    assert metric(scores, test_matrix, k=K) == pytest.approx(
        metric(RECOMMENDED, test_matrix, k=K)
    )


@pytest.mark.parametrize("metric", METRICS, ids=METRIC_IDS)
def test_users_without_test_items_are_skipped(metric, test_matrix):
    """Adding users with an empty test row must not move the numbers."""
    padded_test = sparse.vstack(
        [test_matrix, sparse.csr_matrix((4, N_ITEMS))], format="csr"
    )
    padded_recs = np.vstack([RECOMMENDED, np.zeros((4, K), dtype=int)])

    assert metric(padded_recs, padded_test, k=K) == pytest.approx(
        metric(RECOMMENDED, test_matrix, k=K)
    )


@pytest.mark.parametrize("metric", METRICS, ids=METRIC_IDS)
def test_no_evaluable_users_gives_zero(metric):
    assert metric(RECOMMENDED, sparse.csr_matrix((3, N_ITEMS)), k=K) == 0.0


@pytest.mark.parametrize("metric", METRICS, ids=METRIC_IDS)
def test_ranking_with_no_hits_scores_zero(metric, test_matrix):
    # User 2 has only two non-held-out items, so its third slot repeats one;
    # duplicates are irrelevant here because neither is a hit.
    nothing = np.array([[0, 2, 4], [0, 1, 2], [1, 3, 3]])
    assert metric(nothing, test_matrix, k=K) == 0.0


@pytest.mark.parametrize(
    ("metric", "expected"), list(zip(METRICS, PERFECT, strict=True)), ids=METRIC_IDS
)
def test_perfect_ranking(metric, expected, test_matrix):
    """Held-out items first: recall and NDCG reach 1, precision cannot."""
    perfect = np.array([[1, 3, 4], [0, 1, 2], [0, 2, 4]])
    assert metric(perfect, test_matrix, k=K) == pytest.approx(expected)


def test_ndcg_rewards_putting_hits_first(test_matrix):
    early = np.array([[1, 3, 0], [0, 0, 0], [0, 2, 4]])
    late = np.array([[0, 1, 3], [0, 0, 0], [1, 0, 2]])
    assert metrics.ndcg_at_k(early, test_matrix, k=K) > metrics.ndcg_at_k(
        late, test_matrix, k=K
    )


def test_k_larger_than_the_score_matrix_is_clipped(scores, test_matrix):
    """Only ``N_ITEMS`` items exist, so k=10 can rank at most that many."""
    assert metrics.recall_at_k(scores, test_matrix, k=10) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        (RECOMMENDED, "need at least k="),
        (RECOMMENDED[:2], "but predictions cover"),
    ],
    ids=["too_few_ranks", "wrong_user_count"],
)
def test_malformed_predictions_are_reported(predictions, message, test_matrix):
    k = 4 if message.startswith("need") else K
    with pytest.raises(ValueError, match=message):
        metrics.precision_at_k(predictions, test_matrix, k=k)


@pytest.mark.parametrize("metric", METRICS, ids=METRIC_IDS)
def test_metrics_accept_fastslim_output(metric, fitted):
    """End-to-end: fit, score, rank, measure -- shapes and ranges line up."""
    history, weights = fitted
    # Score against the training interactions themselves, so there is something
    # to hit; masking them would make every metric trivially zero.
    scores = fastslim.predict(weights, history, exclude_seen=False)
    top = fastslim.recommend(weights, history, k=2, exclude_seen=False)

    from_scores = metric(scores, history, k=2)
    assert 0.0 < from_scores <= 1.0
    assert from_scores == pytest.approx(metric(top, history, k=2))
