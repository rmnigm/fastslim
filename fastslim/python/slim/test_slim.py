"""Tests for SLIM recommender system."""

import numpy as np
from scipy import sparse

import slim


class TestFit:
    def test_basic_fit(self):
        np.random.seed(42)

        interactions = sparse.random(50, 30, density=0.1, format="csr")
        weights = slim.fit(interactions, lambd=0.5, beta=0.5, max_iter=50)

        assert weights.shape == (30, 30)
        assert sparse.issparse(weights)
        assert np.allclose(weights.diagonal(), 0.0)
        assert np.all(weights.data >= 0)

    def test_input_conversion(self):
        np.random.seed(42)

        csr = sparse.random(20, 15, density=0.1, format="csr")
        coo = csr.tocoo()
        csc = csr.tocsc()

        w1 = slim.fit(csr, lambd=0.5, beta=0.5, max_iter=20)
        w2 = slim.fit(coo, lambd=0.5, beta=0.5, max_iter=20)
        w3 = slim.fit(csc, lambd=0.5, beta=0.5, max_iter=20)

        np.testing.assert_array_almost_equal(w1.toarray(), w2.toarray())
        np.testing.assert_array_almost_equal(w1.toarray(), w3.toarray())

    def test_empty_matrix(self):
        interactions = sparse.csr_matrix((10, 5))
        weights = slim.fit(interactions, lambd=0.5, beta=0.5)

        assert weights.shape == (5, 5)
        assert weights.nnz == 0, "Empty input should produce empty weights"


class TestIntegration:
    def test_full_pipeline(self):
        # TODO test with normal data
        pass
