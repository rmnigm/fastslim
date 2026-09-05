"""The solver promises bit-for-bit reproducibility, including across threads."""

from __future__ import annotations

import fastslim
import numpy as np
import pytest
from fastslim import _slim_rs

PARAMS = {"lambd": 0.4, "beta": 0.4, "tol": 1e-10, "max_iter": 1000}


def fingerprint(W) -> tuple[bytes, bytes, bytes]:
    """Raw bytes of the three CSR arrays -- no tolerance, no rounding."""
    return W.indptr.tobytes(), W.indices.tobytes(), W.data.tobytes()


def test_repeated_runs_are_identical(binary_matrix):
    first = fastslim.fit(binary_matrix, **PARAMS)
    for _ in range(3):
        assert fingerprint(fastslim.fit(binary_matrix, **PARAMS)) == fingerprint(first)


@pytest.mark.parametrize("n_threads", [1, 3, None])
def test_thread_count_does_not_change_the_answer(make_binary, n_threads):
    X = make_binary(n_users=400, n_items=60, density=0.1, seed=7)
    reference = fastslim.fit(X, n_threads=1, **PARAMS)
    assert fingerprint(fastslim.fit(X, n_threads=n_threads, **PARAMS)) == fingerprint(
        reference
    )


def test_int32_and_int64_index_arrays_agree(make_binary):
    """scipy picks the index dtype for us, so go straight to the binding."""
    X = make_binary(n_users=120, n_items=25, density=0.15, seed=8)
    n_rows, n_cols = X.shape

    results = [
        _slim_rs.solve_slim(
            data=X.data,
            indices=X.indices.astype(dtype),
            indptr=X.indptr.astype(dtype),
            n_rows=n_rows,
            n_cols=n_cols,
            **PARAMS,
        )
        for dtype in (np.int32, np.int64)
    ]

    assert results[0][2].size > 0
    for from_i32, from_i64 in zip(*results, strict=True):
        assert from_i32.tobytes() == from_i64.tobytes()


def test_solver_returns_the_documented_csc_layout(make_binary):
    """Segment ``i`` of the output holds the neighbours of target item ``i``."""
    X = make_binary(n_users=120, n_items=25, density=0.15, seed=8)
    indptr, indices, data = _slim_rs.solve_slim(
        data=X.data,
        indices=X.indices,
        indptr=X.indptr,
        n_rows=X.shape[0],
        n_cols=X.shape[1],
        **PARAMS,
    )

    assert indptr.dtype == np.int64
    assert indices.dtype == np.int64
    assert data.dtype == np.float64
    assert indptr.shape == (X.shape[1] + 1,)
    assert indptr[0] == 0 and indptr[-1] == data.size
    assert np.all(data > 0)
    for target, (start, stop) in enumerate(zip(indptr[:-1], indptr[1:], strict=True)):
        neighbours = indices[start:stop]
        assert np.all(np.diff(neighbours) > 0), "neighbours must be sorted, no dupes"
        assert target not in neighbours, "an item is never its own neighbour"
