"""``ConvergenceWarning`` and the per-item diagnostics on the estimator."""

from __future__ import annotations

import warnings

import fastslim
import numpy as np
import pytest
from fastslim import SLIM, ConvergenceWarning
from scipy import sparse

PARAMS = {"lambd": 0.1, "beta": 0.5}
GENEROUS = 10_000


@pytest.fixture
def ill_conditioned():
    """Two identical popular columns make coordinate descent crawl.

    For target item 2 the candidates 0 and 1 have identical columns, so the
    objective is flat along ``w_0 - w_1`` except for the ``beta`` term and each
    pass shrinks the error only by ``(P / (P + beta)) ** 2``: hundreds of
    passes at the default ``tol``, far more than a budget of three.
    """
    X = np.zeros((60, 4))
    X[:50, 0] = 1.0
    X[:50, 1] = 1.0
    X[:40, 2] = 1.0
    X[10:30, 3] = 1.0
    return sparse.csr_matrix(X)


def test_warning_fires_when_the_budget_is_too_small(ill_conditioned):
    with pytest.warns(
        ConvergenceWarning, match=r"\d+ of 4 items did not converge within max_iter=3"
    ):
        fastslim.fit(ill_conditioned, max_iter=3, **PARAMS)


def test_no_warning_with_a_generous_budget(ill_conditioned):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        fastslim.fit(ill_conditioned, max_iter=GENEROUS, **PARAMS)
        SLIM(max_iter=GENEROUS, **PARAMS).fit(ill_conditioned)


def test_warning_points_at_the_caller(ill_conditioned):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fastslim.fit(ill_conditioned, max_iter=3, **PARAMS)
        SLIM(max_iter=3, **PARAMS).fit(ill_conditioned)
    assert [w.category for w in caught] == [ConvergenceWarning, ConvergenceWarning]
    assert all(w.filename == __file__ for w in caught)


def test_estimator_records_per_item_diagnostics(ill_conditioned):
    with pytest.warns(ConvergenceWarning):
        truncated = SLIM(max_iter=3, **PARAMS).fit(ill_conditioned)
    assert truncated.n_passes_.dtype == np.int64
    assert truncated.converged_.dtype == np.bool_
    assert truncated.n_passes_.shape == truncated.converged_.shape == (4,)
    assert not truncated.converged_[2]
    assert truncated.n_passes_[2] == 3
    assert np.all(truncated.n_passes_ <= 3)

    full = SLIM(max_iter=GENEROUS, **PARAMS).fit(ill_conditioned)
    assert full.converged_.all()
    assert np.all(full.n_passes_ <= GENEROUS)
    assert full.n_passes_[2] > 3


def test_truncated_and_converged_weights_differ(ill_conditioned):
    with pytest.warns(ConvergenceWarning):
        truncated = fastslim.fit(ill_conditioned, max_iter=3, **PARAMS)
    full = fastslim.fit(ill_conditioned, max_iter=GENEROUS, tol=1e-8, **PARAMS)
    assert not np.allclose(truncated.toarray(), full.toarray())
    # At the optimum the two identical columns share the weight equally.
    assert full[0, 2] == pytest.approx(full[1, 2], abs=1e-6)


def test_max_iter_zero_warns_when_there_is_work_to_do(binary_matrix):
    with pytest.warns(ConvergenceWarning, match="max_iter=0"):
        W = fastslim.fit(binary_matrix, max_iter=0)
    assert W.nnz == 0


def test_no_warning_for_an_empty_problem():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        W = fastslim.fit(sparse.csr_matrix((5, 3)), max_iter=0)
    assert W.nnz == 0


def test_warning_is_a_user_warning_and_exported():
    assert issubclass(ConvergenceWarning, UserWarning)
    assert "ConvergenceWarning" in fastslim.__all__
