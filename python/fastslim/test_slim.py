"""Tests for SLIM recommender system."""

import numpy as np
from scipy import sparse
from implicit.datasets.movielens import get_movielens
from implicit.evaluation import train_test_split

import fastslim


class TestFit:
    def test_basic_fit(self):
        np.random.seed(42)

        interactions = sparse.random(50, 30, density=0.1, format="csr")
        weights = fastslim.fit(interactions, lambd=0.5, beta=0.5, max_iter=50)

        assert weights.shape == (30, 30)
        assert sparse.issparse(weights)
        assert np.allclose(weights.diagonal(), 0.0)
        assert np.all(weights.data >= 0)

    def test_input_conversion(self):
        np.random.seed(42)

        csr = sparse.random(20, 15, density=0.1, format="csr")
        coo = csr.tocoo()
        csc = csr.tocsc()

        w1 = fastslim.fit(csr, lambd=0.5, beta=0.5, max_iter=20)
        w2 = fastslim.fit(coo, lambd=0.5, beta=0.5, max_iter=20)
        w3 = fastslim.fit(csc, lambd=0.5, beta=0.5, max_iter=20)

        np.testing.assert_array_almost_equal(w1.toarray(), w2.toarray())
        np.testing.assert_array_almost_equal(w1.toarray(), w3.toarray())

    def test_empty_matrix(self):
        interactions = sparse.csr_matrix((10, 5))
        weights = fastslim.fit(interactions, lambd=0.5, beta=0.5)

        assert weights.shape == (5, 5)
        assert weights.nnz == 0, "Empty input should produce empty weights"


class TestIntegration:
    def test_full_pipeline(self):
        _, ratings = get_movielens("100k")
        user_item = ratings.T.tocsr()
        user_item = (user_item > 0).astype(np.float64)

        train, test = train_test_split(user_item, train_percentage=0.8, random_state=42)

        weights = fastslim.fit(train, lambd=0.1, beta=0.1, max_iter=10)

        rng = np.random.default_rng(42)
        user_indices = np.arange(train.shape[0])
        rng.shuffle(user_indices)
        user_indices = user_indices[:50]

        recalls = []
        k = 10
        for user_idx in user_indices:
            test_items = test[user_idx].indices
            if test_items.size == 0:
                continue
            history = train[user_idx].tocsr()
            scores = fastslim.predict(weights, history, exclude_seen=True)
            assert scores.shape == (train.shape[1],)
            assert np.isfinite(scores).any()
            assert np.all(np.isneginf(scores[history.indices]))
            top_k = np.argpartition(scores, -k)[-k:]
            recalls.append(len(set(top_k) & set(test_items)) / len(test_items))

        assert recalls, "Expected at least one user with test interactions"
        assert float(np.mean(recalls)) >= 0.15
