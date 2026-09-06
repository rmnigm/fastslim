from __future__ import annotations

import fastslim
import numpy as np
import pytest
from conftest import TINY
from fastslim import SLIM, NotFittedError
from scipy import sparse

PARAMS = {"lambd": 0.3, "beta": 0.3, "max_iter": 200, "tol": 1e-9}


@pytest.fixture
def model(tiny_matrix) -> SLIM:
    """A ``SLIM`` fitted on ``tiny_matrix`` with ``PARAMS``."""
    return SLIM(**PARAMS).fit(tiny_matrix)


def test_fit_returns_self(tiny_matrix):
    unfitted = SLIM()
    assert unfitted.fit(tiny_matrix) is unfitted


def test_fit_matches_the_function(model, tiny_matrix):
    expected = fastslim.fit(tiny_matrix, **PARAMS)
    for name in ("indptr", "indices", "data"):
        np.testing.assert_array_equal(
            getattr(model.weights, name), getattr(expected, name)
        )
    assert model.n_items == tiny_matrix.shape[1]
    assert isinstance(model.n_items, int)


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("predict", {}),
        ("predict", {"exclude_seen": False}),
        ("recommend", {"k": 2}),
        ("recommend", {"k": 2, "batch_size": 1}),
    ],
    ids=["predict", "predict_all", "recommend", "recommend_batched"],
)
def test_predict_and_recommend_match_the_functions(model, tiny_matrix, method, kwargs):
    from_function = getattr(fastslim, method)(model.weights, tiny_matrix, **kwargs)
    np.testing.assert_array_equal(
        getattr(model, method)(tiny_matrix, **kwargs), from_function
    )


def test_y_is_ignored(tiny_matrix):
    with_y = SLIM(**PARAMS).fit(tiny_matrix, y=np.arange(tiny_matrix.shape[0]))
    np.testing.assert_array_equal(
        with_y.weights.data, SLIM(**PARAMS).fit(tiny_matrix).weights.data
    )


@pytest.mark.parametrize("method", ["predict", "recommend"])
def test_not_fitted_error(method, tiny_matrix):
    with pytest.raises(NotFittedError, match="not fitted yet"):
        getattr(SLIM(), method)(tiny_matrix)


@pytest.mark.parametrize("caught", [ValueError, AttributeError])
def test_not_fitted_error_is_both_value_and_attribute_error(caught):
    assert issubclass(NotFittedError, caught)
    with pytest.raises(caught, match="not fitted yet"):
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
    unfitted = SLIM()
    assert unfitted.set_params(lambd=2.0) is unfitted
    assert unfitted.lambd == 2.0
    unfitted.fit(tiny_matrix)
    assert unfitted.weights.nnz == fastslim.fit(tiny_matrix, lambd=2.0).nnz


@pytest.mark.parametrize(
    ("estimator", "expected"),
    [
        (SLIM(), "SLIM()"),
        (SLIM(lambd=1.0), "SLIM(lambd=1.0)"),
        (SLIM(lambd=1.0, n_threads=4), "SLIM(lambd=1.0, n_threads=4)"),
    ],
    ids=["defaults", "one", "two"],
)
def test_repr_shows_only_non_defaults(estimator, expected):
    assert repr(estimator) == expected


def test_repr_rebuilds_an_equivalent_estimator():
    original = SLIM(beta=0.125, max_iter=3)
    assert eval(repr(original)).get_params() == original.get_params()  # noqa: S307


def test_refitting_replaces_the_weights(tiny_matrix):
    refitted = SLIM(lambd=0.1).fit(tiny_matrix)
    assert refitted.weights.nnz > 0
    refitted.set_params(lambd=1e6).fit(tiny_matrix)
    assert refitted.weights.nnz == 0


@pytest.mark.parametrize(
    ("convert", "container"),
    [(sparse.csr_matrix, sparse.csr_matrix), (sparse.csr_array, sparse.csr_array)],
    ids=["matrix", "array"],
)
def test_weights_container_mirrors_the_input(convert, container):
    assert isinstance(SLIM().fit(convert(TINY)).weights, container)
