"""End-to-end quality check on MovieLens 100k.

Marked ``slow`` because the first run downloads the dataset to
``~/implicit_data`` (a few MB); afterwards the whole test takes about a second.
Skipped entirely when ``implicit`` is not installed -- it is in the ``bench``
dependency group, not ``dev``.
"""

from __future__ import annotations

import fastslim
import numpy as np
import pytest
from fastslim import metrics

pytest.importorskip("implicit", reason="install the 'bench' group for this test")

from implicit.datasets.movielens import get_movielens  # noqa: E402
from implicit.evaluation import train_test_split  # noqa: E402

# ``max_iter=50`` deliberately truncates the ill-conditioned popular items (it
# is the setting the benchmark uses); the metrics below are what is guarded,
# so the ConvergenceWarning that truncation raises is expected here.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.filterwarnings("ignore::fastslim.ConvergenceWarning"),
]

K = 10
FIT_PARAMS = {"lambd": 2.0, "beta": 2.0, "max_iter": 50}

# Measured on 2026-09-05 with this exact split (implicit's train_test_split at
# random_state=42, 80/20) and these hyperparameters:
#
#     precision@10 = 0.3391   recall@10 = 0.2236   ndcg@10 = 0.4069
#
# The thresholds below sit 0.01 under the measured values, which is far wider
# than any run-to-run noise (there is none -- everything here is seeded) but
# tight enough to catch a real regression in the solver or the metrics.
EXPECTED_RECALL = 0.2236
EXPECTED_NDCG = 0.4069
TOLERANCE = 0.01


@pytest.fixture(scope="module")
def movielens_split():
    _, ratings = get_movielens("100k")
    # get_movielens returns (items x users); SLIM wants (users x items), and
    # implicit feedback means "rated at all", not "rated highly".
    user_item = (ratings.T.tocsr() > 0).astype(np.float64)
    train, test = train_test_split(user_item, train_percentage=0.8, random_state=42)
    return train.tocsr(), test.tocsr()


def test_recall_and_ndcg_hold_up(movielens_split):
    train, test = movielens_split

    weights = fastslim.fit(train, **FIT_PARAMS)
    assert weights.shape == (train.shape[1], train.shape[1])
    assert weights.nnz > 0

    scores = fastslim.predict(weights, train, batch_size=500)
    assert scores.shape == train.shape

    recall = metrics.recall_at_k(scores, test, k=K)
    ndcg = metrics.ndcg_at_k(scores, test, k=K)

    assert recall >= EXPECTED_RECALL - TOLERANCE, f"recall@{K} regressed to {recall}"
    assert ndcg >= EXPECTED_NDCG - TOLERANCE, f"ndcg@{K} regressed to {ndcg}"


def test_recommend_agrees_with_scoring(movielens_split):
    """The ranked output must score the same as the raw score matrix."""
    train, test = movielens_split
    weights = fastslim.fit(train, **FIT_PARAMS)

    scores = fastslim.predict(weights, train, batch_size=500)
    top = fastslim.recommend(weights, train, k=K, batch_size=500)

    assert top.shape == (train.shape[0], K)
    assert metrics.recall_at_k(top, test, k=K) == pytest.approx(
        metrics.recall_at_k(scores, test, k=K)
    )
    # Nothing already in the training history may be recommended.
    seen = np.repeat(np.arange(train.shape[0]), np.diff(train.indptr))
    seen_keys = set(zip(seen.tolist(), train.indices.tolist(), strict=True))
    recommended_keys = {
        (user, int(item)) for user, row in enumerate(top) for item in row
    }
    assert not (seen_keys & recommended_keys)


def test_estimator_reproduces_the_functional_pipeline(movielens_split):
    train, _ = movielens_split
    model = fastslim.SLIM(**FIT_PARAMS).fit(train)
    expected = fastslim.fit(train, **FIT_PARAMS)
    np.testing.assert_array_equal(model.weights_.data, expected.data)
