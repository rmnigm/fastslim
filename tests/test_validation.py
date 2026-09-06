from __future__ import annotations

import fastslim
import numpy as np
import pytest
from fastslim import SLIM, native
from scipy import sparse

GOOD = sparse.csr_matrix(np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]))


def with_value(value: float) -> sparse.csr_matrix:
    """``GOOD`` with its second stored entry replaced by ``value``."""
    matrix = GOOD.copy()
    matrix.data[1] = value
    return matrix


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.nan, "finite"),
        (np.inf, "finite"),
        (-np.inf, "finite"),
        (-1.0, "non-negative"),
    ],
)
def test_bad_data_values_name_the_offending_entry(value, message):
    with pytest.raises(ValueError, match=message) as excinfo:
        fastslim.fit(with_value(value))
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
    ("kwargs", "exc", "message"),
    [
        ({"lambd": -1}, ValueError, "lambd must be >= 0"),
        ({"max_iter": 1.5}, TypeError, "max_iter must be an integer"),
        ({"n_threads": 0}, ValueError, "n_threads must be >= 1"),
    ],
)
def test_estimator_validates_at_fit_time(kwargs, exc, message):
    """Follow the sklearn convention: the constructor stores, ``fit`` checks."""
    name, value = next(iter(kwargs.items()))
    model = SLIM(**kwargs)
    assert model.get_params()[name] == value
    with pytest.raises(exc, match=message):
        model.fit(GOOD)


def test_numpy_scalar_hyperparameters_are_accepted():
    weights = fastslim.fit(
        GOOD,
        lambd=np.float32(0.5),
        beta=np.float64(0.5),
        max_iter=np.int64(10),
        n_threads=np.int32(1),
    )
    assert weights.shape == (3, 3)


@pytest.mark.parametrize(
    ("history", "message"),
    [
        (np.ones(5), "items but weights describe"),
        (np.ones((2, 2, 3)), "must be 1-D or 2-D"),
    ],
    ids=["wrong_width", "3d"],
)
def test_predict_rejects_bad_history(history, message):
    with pytest.raises(ValueError, match=message):
        fastslim.predict(fastslim.fit(GOOD), history)


@pytest.mark.parametrize(
    ("kwargs", "exc", "message"),
    [
        ({"k": 0}, ValueError, "k must be >= 1"),
        ({"k": -1}, ValueError, "k must be >= 1"),
        ({"k": 1.5}, TypeError, "k must be an integer"),
        ({"batch_size": 0}, ValueError, "batch_size must be >= 1"),
    ],
)
def test_recommend_rejects_bad_k_and_batch_size(kwargs, exc, message):
    with pytest.raises(exc, match=message):
        fastslim.recommend(fastslim.fit(GOOD), GOOD, **kwargs)


def test_set_params_rejects_unknown_names():
    with pytest.raises(ValueError, match="invalid parameter 'alpha'"):
        SLIM().set_params(alpha=1.0)


def test_binding_rejects_malformed_indptr():
    """The Rust layer keeps its own guardrails; check one reaches Python."""
    with pytest.raises(ValueError, match="indptr"):
        native.solve_slim(
            data=np.array([1.0, 1.0]),
            indices=np.array([0, 1], dtype=np.int32),
            indptr=np.array([0, 2, 1], dtype=np.int32),
            n_rows=2,
            n_cols=2,
        )
