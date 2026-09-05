"""Bad input must fail loudly, early, and with a message that names the problem."""

from __future__ import annotations

import fastslim
import numpy as np
import pytest
from fastslim import SLIM, _slim_rs
from scipy import sparse

GOOD = sparse.csr_matrix(np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]))


def _with_value(value: float) -> sparse.csr_matrix:
    X = sparse.csr_matrix(np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]))
    X.data[1] = value
    return X


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.nan, "finite"),
        (np.inf, "finite"),
        (-np.inf, "finite"),
        (-1.0, "non-negative"),
    ],
)
def test_bad_data_values(value, message):
    with pytest.raises(ValueError, match=message) as excinfo:
        fastslim.fit(_with_value(value))
    # The message points at the offending entry, not just at the array.
    assert "row 0, column 1" in str(excinfo.value)


def test_negative_dense_input():
    with pytest.raises(ValueError, match="non-negative"):
        fastslim.fit(np.array([[1.0, -2.0], [0.0, 1.0]]))


@pytest.mark.parametrize(
    "matrix",
    [
        pytest.param(np.ones(4), id="1d_dense"),
        pytest.param(np.ones((2, 3, 4)), id="3d_dense"),
        pytest.param(sparse.csr_array(np.ones(4)), id="1d_sparse_array"),
    ],
)
def test_wrong_dimensionality(matrix):
    with pytest.raises(ValueError, match="must be 2-D"):
        fastslim.fit(matrix)


@pytest.mark.parametrize(
    ("kwargs", "exc", "message"),
    [
        ({"lambd": -1}, ValueError, "lambd must be >= 0"),
        ({"lambd": np.nan}, ValueError, "lambd must be finite"),
        ({"lambd": "0.5"}, TypeError, "lambd must be a real number"),
        ({"beta": -1}, ValueError, "beta must be >= 0"),
        ({"beta": None}, TypeError, "beta must be a real number"),
        ({"max_iter": -1}, ValueError, "max_iter must be >= 0"),
        ({"max_iter": 1.5}, TypeError, "max_iter must be an integer"),
        ({"max_iter": True}, TypeError, "max_iter must be an integer"),
        ({"tol": -1}, ValueError, "tol must be >= 0"),
        ({"n_threads": 0}, ValueError, "n_threads must be >= 1"),
        ({"n_threads": -1}, ValueError, "n_threads must be >= 1"),
        ({"n_threads": 1.5}, TypeError, "n_threads must be an integer"),
    ],
)
def test_bad_hyperparameters(kwargs, exc, message):
    with pytest.raises(exc, match=message):
        fastslim.fit(GOOD, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"lambd": -1}, ValueError),
        ({"max_iter": 1.5}, TypeError),
        ({"n_threads": 0}, ValueError),
    ],
)
def test_estimator_validates_at_fit_time(kwargs, exc):
    """sklearn convention: the constructor stores, ``fit`` checks."""
    name, value = next(iter(kwargs.items()))
    model = SLIM(**kwargs)
    assert model.get_params()[name] == value
    with pytest.raises(exc):
        model.fit(GOOD)


def test_numpy_scalar_hyperparameters_are_accepted():
    W = fastslim.fit(
        GOOD,
        lambd=np.float32(0.5),
        beta=np.float64(0.5),
        max_iter=np.int64(10),
        n_threads=np.int32(1),
    )
    assert W.shape == (3, 3)


def test_predict_rejects_mismatched_history():
    W = fastslim.fit(GOOD)
    with pytest.raises(ValueError, match="items but weights describe"):
        fastslim.predict(W, np.ones(5))


def test_predict_rejects_3d_history():
    W = fastslim.fit(GOOD)
    with pytest.raises(ValueError, match="must be 1-D or 2-D"):
        fastslim.predict(W, np.ones((2, 2, 3)))


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"k": 0}, ValueError),
        ({"k": -1}, ValueError),
        ({"k": 1.5}, TypeError),
        ({"batch_size": 0}, ValueError),
    ],
)
def test_recommend_rejects_bad_k_and_batch_size(kwargs, exc):
    W = fastslim.fit(GOOD)
    with pytest.raises(exc):
        fastslim.recommend(W, GOOD, **kwargs)


def test_set_params_rejects_unknown_names():
    with pytest.raises(ValueError, match="invalid parameter 'alpha'"):
        SLIM().set_params(alpha=1.0)


def test_binding_rejects_malformed_indptr():
    """The Rust layer keeps its own guardrails; check one reaches Python."""
    with pytest.raises(ValueError, match="indptr"):
        _slim_rs.solve_slim(
            data=np.array([1.0, 1.0]),
            indices=np.array([0, 1], dtype=np.int32),
            indptr=np.array([0, 2, 1], dtype=np.int32),
            n_rows=2,
            n_cols=2,
        )
