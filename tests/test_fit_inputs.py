"""Every accepted way of spelling the same matrix must give the same model."""

from __future__ import annotations

import fastslim
import numpy as np
import pytest
from conftest import to_dense
from scipy import sparse

PARAMS = {"lambd": 0.3, "beta": 0.3, "tol": 1e-10, "max_iter": 1000}

DENSE = np.array(
    [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ]
)


@pytest.fixture
def baseline() -> np.ndarray:
    return to_dense(fastslim.fit(sparse.csr_matrix(DENSE), **PARAMS))


@pytest.mark.parametrize(
    "convert",
    [
        pytest.param(sparse.csr_matrix, id="csr"),
        pytest.param(sparse.csc_matrix, id="csc"),
        pytest.param(sparse.coo_matrix, id="coo"),
        pytest.param(sparse.lil_matrix, id="lil"),
        pytest.param(sparse.dok_matrix, id="dok"),
        pytest.param(sparse.bsr_matrix, id="bsr"),
        pytest.param(sparse.csr_array, id="csr_array"),
        pytest.param(sparse.coo_array, id="coo_array"),
        pytest.param(np.asarray, id="ndarray"),
        pytest.param(lambda a: a.tolist(), id="list_of_lists"),
        pytest.param(
            np.matrix,
            id="np_matrix",
            marks=pytest.mark.filterwarnings("ignore::PendingDeprecationWarning"),
        ),
    ],
)
def test_all_containers_agree(baseline, convert):
    assert np.allclose(to_dense(fastslim.fit(convert(DENSE), **PARAMS)), baseline)


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64, np.bool_, np.float32])
def test_dtypes_agree(baseline, dtype):
    X = sparse.csr_matrix(DENSE.astype(dtype))
    assert np.allclose(to_dense(fastslim.fit(X, **PARAMS)), baseline)


def _messy_csr() -> sparse.csr_matrix:
    """A deliberately non-canonical CSR spelling of ``DENSE``.

    Row 0 of ``DENSE`` is ``[1, 1, 0, 0]``; here column 1 is split into
    ``0.6 + 0.4``, the columns are listed out of order, and column 3 carries an
    explicit zero.  Every other row is spelled normally.
    """
    data: list[float] = []
    indices: list[int] = []
    indptr: list[int] = [0]
    for row_index, row in enumerate(DENSE):
        if row_index == 0:
            data += [0.6, 1.0, 0.4, 0.0]
            indices += [1, 0, 1, 3]
        else:
            columns = np.flatnonzero(row)
            data += row[columns].tolist()
            indices += columns.tolist()
        indptr.append(len(data))
    return sparse.csr_matrix(
        (np.array(data), np.array(indices), np.array(indptr)), shape=DENSE.shape
    )


def test_duplicates_unsorted_indices_and_explicit_zeros(baseline):
    messy = _messy_csr()
    assert not messy.has_canonical_format
    np.testing.assert_allclose(to_dense(messy), DENSE)

    assert np.allclose(to_dense(fastslim.fit(messy, **PARAMS)), baseline)


def test_input_is_not_mutated():
    """Canonicalisation must never write back into the caller's arrays."""
    messy = _messy_csr()
    data = messy.data.copy()
    indices = messy.indices.copy()
    indptr = messy.indptr.copy()

    fastslim.fit(messy, **PARAMS)

    np.testing.assert_array_equal(messy.data, data)
    np.testing.assert_array_equal(messy.indices, indices)
    np.testing.assert_array_equal(messy.indptr, indptr)


def test_explicit_zeros_do_not_count_as_interactions():
    """An explicit zero must behave exactly like a structural one."""
    clean = sparse.csr_matrix(DENSE)
    # Insert an explicit zero at (0, 2); row 0 already ends after two entries,
    # so column order stays ascending and only ``has_canonical_format`` differs.
    with_zero = sparse.csr_matrix(
        (
            np.insert(clean.data, 2, 0.0),
            np.insert(clean.indices, 2, 2),
            np.concatenate([clean.indptr[:1], clean.indptr[1:] + 1]),
        ),
        shape=clean.shape,
    )
    np.testing.assert_allclose(to_dense(with_zero), DENSE)

    assert np.allclose(
        to_dense(fastslim.fit(with_zero, **PARAMS)),
        to_dense(fastslim.fit(clean, **PARAMS)),
    )


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((10, 5), (5, 5)),
        ((0, 5), (5, 5)),
        ((5, 0), (0, 0)),
        ((1, 4), (4, 4)),
        ((4, 1), (1, 1)),
        ((0, 0), (0, 0)),
    ],
)
def test_degenerate_shapes(shape, expected):
    W = fastslim.fit(sparse.csr_matrix(shape), **PARAMS)
    assert W.shape == expected
    assert W.nnz == 0


def test_single_user():
    W = fastslim.fit(sparse.csr_matrix(np.array([[1.0, 1.0, 1.0]])), **PARAMS)
    assert W.shape == (3, 3)
    # One user contributes P = all-ones, so every off-diagonal weight is the
    # same positive number; nothing degenerate happens.
    assert W.nnz == 6
    assert np.all(W.data > 0)


def test_single_item():
    W = fastslim.fit(sparse.csr_matrix(np.ones((5, 1))), **PARAMS)
    assert W.shape == (1, 1)
    assert W.nnz == 0


def test_all_zero_rows_and_columns():
    dense = DENSE.copy()
    dense[2] = 0.0  # a user with no interactions
    dense[:, 1] = 0.0  # an item nobody touched
    W = fastslim.fit(sparse.csr_matrix(dense), **PARAMS)

    assert W.shape == (4, 4)
    # The dead item can neither be a target nor a neighbour.
    assert W[1].nnz == 0
    assert W[:, 1].nnz == 0
    assert W.nnz > 0


def test_sparse_array_in_sparse_array_out():
    W = fastslim.fit(sparse.csr_array(DENSE), **PARAMS)
    assert isinstance(W, sparse.csr_array)
    assert isinstance(W, sparse.sparray)


@pytest.mark.parametrize(
    "convert",
    [sparse.csr_matrix, sparse.coo_matrix, np.asarray, lambda a: a.tolist()],
)
def test_non_sparse_array_input_gives_csr_matrix(convert):
    W = fastslim.fit(convert(DENSE), **PARAMS)
    assert isinstance(W, sparse.csr_matrix)
    assert not isinstance(W, sparse.sparray)


def test_output_is_canonical_csr(binary_matrix):
    W = fastslim.fit(binary_matrix, **PARAMS)
    assert W.format == "csr"
    assert W.has_sorted_indices
    assert W.has_canonical_format
    # Verify independently of scipy's bookkeeping flags.
    for start, stop in zip(W.indptr[:-1], W.indptr[1:], strict=True):
        row = W.indices[start:stop]
        assert np.all(np.diff(row) > 0)
