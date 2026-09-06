from __future__ import annotations

import fastslim
import numpy as np
import pytest
from conftest import TINY, to_dense
from scipy import sparse

PARAMS = {"lambd": 0.3, "beta": 0.3, "tol": 1e-10, "max_iter": 1000}


@pytest.fixture
def baseline() -> np.ndarray:
    """Weights fitted from the canonical CSR spelling of ``TINY``."""
    return to_dense(fastslim.fit(sparse.csr_matrix(TINY), **PARAMS))


def messy_csr() -> sparse.csr_matrix:
    """``TINY`` with row 0 split into ``0.6 + 0.4``, unsorted, plus a stored zero."""
    data: list[float] = []
    indices: list[int] = []
    indptr: list[int] = [0]
    for row_index, row in enumerate(TINY):
        if row_index == 0:
            data += [0.6, 1.0, 0.4, 0.0]
            indices += [1, 0, 1, 3]
        else:
            columns = np.flatnonzero(row)
            data += row[columns].tolist()
            indices += columns.tolist()
        indptr.append(len(data))
    return sparse.csr_matrix(
        (np.array(data), np.array(indices), np.array(indptr)), shape=TINY.shape
    )


def with_explicit_zero() -> sparse.csr_matrix:
    """``TINY`` with an explicit zero at (0, 2), keeping columns ascending."""
    clean = sparse.csr_matrix(TINY)
    return sparse.csr_matrix(
        (
            np.insert(clean.data, 2, 0.0),
            np.insert(clean.indices, 2, 2),
            np.concatenate([clean.indptr[:1], clean.indptr[1:] + 1]),
        ),
        shape=clean.shape,
    )


def as_dtype(dtype):
    """Build a CSR matrix of ``TINY`` recast to ``dtype``."""
    return lambda a: sparse.csr_matrix(a.astype(dtype))


SPELLINGS = {
    "csr": sparse.csr_matrix,
    "csc": sparse.csc_matrix,
    "coo": sparse.coo_matrix,
    "lil": sparse.lil_matrix,
    "dok": sparse.dok_matrix,
    "bsr": sparse.bsr_matrix,
    "csr_array": sparse.csr_array,
    "coo_array": sparse.coo_array,
    "ndarray": np.asarray,
    "np_matrix": np.matrix,
    "list_of_lists": lambda a: a.tolist(),
    "int8": as_dtype(np.int8),
    "int32": as_dtype(np.int32),
    "int64": as_dtype(np.int64),
    "bool": as_dtype(np.bool_),
    "float32": as_dtype(np.float32),
    "messy": lambda a: messy_csr(),
    "explicit_zero": lambda a: with_explicit_zero(),
}


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.parametrize("convert", SPELLINGS.values(), ids=SPELLINGS)
def test_every_spelling_of_the_same_matrix_agrees(baseline, convert):
    assert np.allclose(to_dense(fastslim.fit(convert(TINY), **PARAMS)), baseline)


@pytest.mark.parametrize(
    ("build", "extra_nnz"),
    [(messy_csr, 2), (with_explicit_zero, 1)],
    ids=["messy", "zero"],
)
def test_awkward_spellings_hold_the_same_numbers(build, extra_nnz):
    """The awkward spellings must really be awkward, or they test nothing."""
    matrix = build()
    np.testing.assert_allclose(to_dense(matrix), TINY)
    assert matrix.nnz == sparse.csr_matrix(TINY).nnz + extra_nnz


def test_messy_spelling_is_not_canonical():
    assert not messy_csr().has_canonical_format


def test_input_is_not_mutated():
    """Canonicalisation must never write back into the caller's arrays."""
    messy = messy_csr()
    before = [messy.data.copy(), messy.indices.copy(), messy.indptr.copy()]

    fastslim.fit(messy, **PARAMS)

    after = [messy.data, messy.indices, messy.indptr]
    for original, current in zip(before, after, strict=True):
        np.testing.assert_array_equal(current, original)


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
    weights = fastslim.fit(sparse.csr_matrix(shape), **PARAMS)
    assert weights.shape == expected
    assert weights.nnz == 0


def test_single_user():
    """One user makes P all-ones, so every off-diagonal weight is the same."""
    weights = fastslim.fit(sparse.csr_matrix(np.ones((1, 3))), **PARAMS)
    assert weights.shape == (3, 3)
    assert weights.nnz == 6
    assert np.all(weights.data > 0)


def test_single_item():
    weights = fastslim.fit(sparse.csr_matrix(np.ones((5, 1))), **PARAMS)
    assert weights.shape == (1, 1)
    assert weights.nnz == 0


def test_all_zero_rows_and_columns():
    """A dead item can be neither a target nor a neighbour."""
    dense = TINY.copy()
    dense[2] = 0.0  # a user with no interactions
    dense[:, 1] = 0.0  # an item nobody touched
    weights = fastslim.fit(sparse.csr_matrix(dense), **PARAMS)

    assert weights.shape == (4, 4)
    assert weights[1].nnz == 0
    assert weights[:, 1].nnz == 0
    assert weights.nnz > 0


@pytest.mark.parametrize(
    ("convert", "is_array"),
    [
        (sparse.csr_array, True),
        (sparse.coo_array, True),
        (sparse.csr_matrix, False),
        (sparse.coo_matrix, False),
        (np.asarray, False),
        (lambda a: a.tolist(), False),
    ],
    ids=["csr_array", "coo_array", "csr_matrix", "coo_matrix", "ndarray", "list"],
)
def test_output_container_mirrors_the_input(convert, is_array):
    weights = fastslim.fit(convert(TINY), **PARAMS)
    assert isinstance(weights, sparse.sparray) is is_array
    assert isinstance(weights, sparse.csr_array if is_array else sparse.csr_matrix)


def test_output_is_canonical_csr(binary_matrix):
    weights = fastslim.fit(binary_matrix, **PARAMS)
    assert weights.format == "csr"
    assert weights.has_canonical_format
    # Verify independently of scipy's bookkeeping flags.
    for start, stop in zip(weights.indptr[:-1], weights.indptr[1:], strict=True):
        assert np.all(np.diff(weights.indices[start:stop]) > 0)
