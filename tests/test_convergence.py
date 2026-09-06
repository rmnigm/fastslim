from __future__ import annotations

import warnings

import fastslim
import numpy as np
import pytest
from fastslim import SLIM, ConvergenceWarning
from scipy import sparse

PARAMS = {"lambd": 0.1, "beta": 0.5}
GENEROUS = 10_000


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
    """``stacklevel`` must name this file, not somewhere inside fastslim."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fastslim.fit(ill_conditioned, max_iter=3, **PARAMS)
        SLIM(max_iter=3, **PARAMS).fit(ill_conditioned)
    assert [w.category for w in caught] == [ConvergenceWarning, ConvergenceWarning]
    assert all(w.filename == __file__ for w in caught)


def test_truncated_fit_reports_per_item_diagnostics(ill_conditioned):
    with pytest.warns(ConvergenceWarning):
        model = SLIM(max_iter=3, **PARAMS).fit(ill_conditioned)
    assert model.n_passes.dtype == np.int64
    assert model.converged.dtype == np.bool_
    assert model.n_passes.shape == model.converged.shape == (4,)
    assert not model.converged[2]
    assert model.n_passes[2] == 3
    assert np.all(model.n_passes <= 3)


def test_converged_fit_reports_per_item_diagnostics(ill_conditioned):
    model = SLIM(max_iter=GENEROUS, **PARAMS).fit(ill_conditioned)
    assert model.converged.all()
    assert np.all(model.n_passes <= GENEROUS)
    assert model.n_passes[2] > 3


def test_truncated_and_converged_weights_differ(ill_conditioned):
    with pytest.warns(ConvergenceWarning):
        truncated = fastslim.fit(ill_conditioned, max_iter=3, **PARAMS)
    full = fastslim.fit(ill_conditioned, max_iter=GENEROUS, tol=1e-8, **PARAMS)
    assert not np.allclose(truncated.toarray(), full.toarray())
    # At the optimum the two identical columns share the weight equally.
    assert full[0, 2] == pytest.approx(full[1, 2], abs=1e-6)


def test_max_iter_zero_warns_and_returns_no_weights(binary_matrix):
    with pytest.warns(ConvergenceWarning, match="max_iter=0"):
        weights = fastslim.fit(binary_matrix, max_iter=0)
    assert weights.nnz == 0


def test_no_warning_for_an_empty_problem():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        weights = fastslim.fit(sparse.csr_matrix((5, 3)), max_iter=0)
    assert weights.nnz == 0


def test_warning_is_a_user_warning_and_exported():
    assert issubclass(ConvergenceWarning, UserWarning)
    assert "ConvergenceWarning" in fastslim.__all__
