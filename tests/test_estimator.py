"""The ``SLIM`` class is a wrapper: it must agree with the functions exactly."""

from __future__ import annotations

import fastslim
import numpy as np
import pytest
from fastslim import SLIM, NotFittedError
from scipy import sparse

PARAMS = {"lambd": 0.3, "beta": 0.3, "max_iter": 200, "tol": 1e-9}


@pytest.fixture
def fitted(tiny_matrix):
    return tiny_matrix, SLIM(**PARAMS).fit(tiny_matrix)


def test_fit_returns_self(tiny_matrix):
    model = SLIM()
    assert model.fit(tiny_matrix) is model


def test_fit_matches_the_function(fitted):
    X, model = fitted
    expected = fastslim.fit(X, **PARAMS)
    np.testing.assert_array_equal(model.weights_.indptr, expected.indptr)
    np.testing.assert_array_equal(model.weights_.indices, expected.indices)
    np.testing.assert_array_equal(model.weights_.data, expected.data)
    assert model.n_items_ == X.shape[1]
    assert isinstance(model.n_items_, int)


def test_fit_sets_the_solver_diagnostic_attributes(fitted):
    """Placeholders until the extension reports per-item convergence."""
    _, model = fitted
    assert hasattr(model, "n_passes_")
    assert hasattr(model, "converged_")


def test_predict_and_recommend_match_the_functions(fitted):
    X, model = fitted
    W = model.weights_
    np.testing.assert_array_equal(model.predict(X), fastslim.predict(W, X))
    np.testing.assert_array_equal(
        model.predict(X, exclude_seen=False),
        fastslim.predict(W, X, exclude_seen=False),
    )
    np.testing.assert_array_equal(
        model.recommend(X, k=2), fastslim.recommend(W, X, k=2)
    )
    np.testing.assert_array_equal(
        model.recommend(X, k=2, batch_size=1), fastslim.recommend(W, X, k=2)
    )


def test_y_is_ignored(tiny_matrix):
    with_y = SLIM(**PARAMS).fit(tiny_matrix, y=np.arange(tiny_matrix.shape[0]))
    without_y = SLIM(**PARAMS).fit(tiny_matrix)
    np.testing.assert_array_equal(with_y.weights_.data, without_y.weights_.data)


@pytest.mark.parametrize("method", ["predict", "recommend"])
def test_not_fitted_error(method, tiny_matrix):
    model = SLIM()
    with pytest.raises(NotFittedError, match="not fitted yet"):
        getattr(model, method)(tiny_matrix)


def test_not_fitted_error_is_both_value_and_attribute_error():
    assert issubclass(NotFittedError, ValueError)
    assert issubclass(NotFittedError, AttributeError)
    with pytest.raises(ValueError):
        SLIM().predict(np.zeros(3))
    with pytest.raises(AttributeError):
        SLIM().predict(np.zeros(3))


def test_get_params_round_trips():
    original = SLIM(lambd=1.5, beta=0.25, max_iter=7, tol=1e-8, n_threads=2)
    params = original.get_params()
    assert params == {
        "lambd": 1.5,
        "beta": 0.25,
        "max_iter": 7,
        "tol": 1e-8,
        "n_threads": 2,
    }
    assert SLIM().set_params(**params).get_params() == params
    assert original.get_params(deep=False) == params


def test_set_params_returns_self_and_takes_effect(tiny_matrix):
    model = SLIM()
    assert model.set_params(lambd=2.0) is model
    assert model.lambd == 2.0
    model.fit(tiny_matrix)
    assert model.weights_.nnz == fastslim.fit(tiny_matrix, lambd=2.0).nnz


def test_repr_shows_only_non_defaults():
    assert repr(SLIM()) == "SLIM()"
    assert repr(SLIM(lambd=1.0)) == "SLIM(lambd=1.0)"
    assert repr(SLIM(lambd=1.0, n_threads=4)) == "SLIM(lambd=1.0, n_threads=4)"
    # The repr is valid Python that rebuilds an equivalent estimator.
    model = SLIM(beta=0.125, max_iter=3)
    assert eval(repr(model)).get_params() == model.get_params()  # noqa: S307


def test_refitting_replaces_the_weights(tiny_matrix):
    model = SLIM(lambd=0.1).fit(tiny_matrix)
    sparse_fit = model.weights_.nnz
    model.set_params(lambd=1e6).fit(tiny_matrix)
    assert sparse_fit > 0
    assert model.weights_.nnz == 0


def test_weights_container_mirrors_the_input(tiny_matrix):
    assert isinstance(SLIM().fit(tiny_matrix).weights_, sparse.csr_matrix)
    assert isinstance(
        SLIM().fit(sparse.csr_array(tiny_matrix)).weights_, sparse.csr_array
    )
