"""Input and parameter validation shared by the functional API and the estimator.

Everything here is private; the public entry points (:func:`fastslim.fit`,
:class:`fastslim.SLIM`) call into it so that both surfaces reject the same
inputs with the same messages.
"""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np
from scipy import sparse

__all__ = [
    "check_float",
    "check_integer",
    "check_interaction_matrix",
    "check_params",
]

# ``sparse.sparray`` only exists on scipy >= 1.11; ``isinstance(x, ())`` is
# always False, which is the right answer on older versions.
SPARRAY: Any = getattr(sparse, "sparray", ())


def check_integer(value: Any, name: str, minimum: int) -> int:
    """Return ``value`` as an ``int``, rejecting non-integers and small values.

    Booleans are rejected even though ``bool`` is a subclass of ``int``: passing
    ``max_iter=True`` is far more likely to be a bug than a request for one pass.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(
            f"{name} must be an integer, got {type(value).__name__} ({value!r})"
        )
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def check_float(value: Any, name: str, minimum: float = 0.0) -> float:
    """Return ``value`` as a finite ``float`` that is at least ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{name} must be a real number, got {type(value).__name__} ({value!r})"
        )
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def check_params(
    lambd: float,
    beta: float,
    max_iter: int,
    tol: float,
    n_threads: int | None,
) -> tuple[float, float, int, float, int | None]:
    """Validate the solver hyperparameters and return them normalised."""
    return (
        check_float(lambd, "lambd"),
        check_float(beta, "beta"),
        check_integer(max_iter, "max_iter", 0),
        check_float(tol, "tol"),
        None if n_threads is None else check_integer(n_threads, "n_threads", 1),
    )


def locate(csr: sparse.csr_matrix, position: int) -> tuple[int, int]:
    """Map an offset into ``csr.data`` back to its ``(row, column)``."""
    row = int(np.searchsorted(csr.indptr, position, side="right") - 1)
    return row, int(csr.indices[position])


def check_values(csr: sparse.csr_matrix, name: str) -> None:
    """Reject non-finite or negative stored values, naming the guilty entry."""
    data = csr.data
    if data.size == 0:
        return
    bad = ~np.isfinite(data)
    if bad.any():
        position = int(np.flatnonzero(bad)[0])
        row, col = locate(csr, position)
        raise ValueError(
            f"{name} must contain only finite values, but the entry at "
            f"row {row}, column {col} is {data[position]}"
        )
    negative = data < 0
    if negative.any():
        position = int(np.flatnonzero(negative)[0])
        row, col = locate(csr, position)
        raise ValueError(
            f"{name} must be non-negative, but the entry at "
            f"row {row}, column {col} is {data[position]}"
        )


def check_interaction_matrix(
    matrix: Any,
    name: str = "interaction_matrix",
) -> tuple[sparse.csr_matrix, bool]:
    """Coerce ``matrix`` to a canonical CSR float64 user-item matrix.

    Accepts any scipy sparse matrix or array as well as dense ``ndarray`` and
    array-likes.  The input is never mutated: copies are made only when a
    conversion, a dtype change or a canonicalisation actually needs one.

    Returns
    -------
    csr : scipy.sparse.csr_matrix or scipy.sparse.csr_array
        Canonical CSR (sorted indices, no duplicates, no explicit zeros) with
        ``float64`` data.
    is_sparse_array : bool
        Whether the input was a scipy *sparse array* (as opposed to a sparse
        matrix or a dense input).  Callers use this to mirror the input type.
    """
    is_sparse_array = isinstance(matrix, SPARRAY)

    if sparse.issparse(matrix):
        if matrix.ndim != 2:
            raise ValueError(
                f"{name} must be 2-D (users x items), got a {matrix.ndim}-D "
                f"{type(matrix).__name__}"
            )
        if matrix.format == "csr":
            csr, owned = matrix, False
        else:
            csr, owned = matrix.tocsr(), True
    else:
        dense = np.asarray(matrix)
        if dense.ndim != 2:
            raise ValueError(
                f"{name} must be 2-D (users x items), got a {dense.ndim}-D array "
                f"with shape {dense.shape}"
            )
        csr, owned = sparse.csr_matrix(dense), True

    if csr.dtype != np.float64:
        csr, owned = csr.astype(np.float64), True

    check_values(csr, name)

    if not csr.has_canonical_format:
        # ``sum_duplicates`` sorts as a side effect, so this covers both
        # unsorted indices and repeated entries.
        if not owned:
            csr, owned = csr.copy(), True
        csr.sum_duplicates()

    if csr.nnz and not csr.data.all():
        # Explicit zeros would otherwise count as "seen" in predict/recommend.
        if not owned:
            csr, owned = csr.copy(), True
        csr.eliminate_zeros()

    return csr, is_sparse_array
