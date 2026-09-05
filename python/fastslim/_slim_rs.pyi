"""Type stubs for the compiled Rust extension ``fastslim._slim_rs``."""

import numpy as np
import numpy.typing as npt

_Indices = npt.NDArray[np.int32] | npt.NDArray[np.int64]

def solve_slim(
    data: npt.NDArray[np.float64],
    indices: _Indices,
    indptr: _Indices,
    n_rows: int,
    n_cols: int,
    lambd: float = ...,
    beta: float = ...,
    max_iter: int = ...,
    tol: float = ...,
    n_threads: int | None = ...,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    r"""Fit SLIM item-item weights on a CSR user-item matrix.

    Solves, independently for each item ``i``,

        min_w  0.5 * ||x_i - X w||^2 + lambd * sum_k w_k
               + (beta / 2) * sum_k w_k^2,     w >= 0, w_i = 0

    by non-negative coordinate descent.  The result is deterministic and does
    not depend on ``n_threads``.

    Parameters
    ----------
    data, indices, indptr
        CSR arrays of the ``(n_rows, n_cols)`` user-item matrix.  ``data`` must
        be ``float64``, finite and non-negative; the index arrays may be
        ``int32`` or ``int64``.  ``indptr`` has ``n_rows + 1`` entries.
    n_rows, n_cols
        Number of users and items.
    lambd, beta
        L1 and L2 penalties, both ``>= 0``.
    max_iter
        Maximum coordinate-descent passes per item; ``0`` gives zero weights.
    tol
        Convergence tolerance on the largest weight change within a pass.
    n_threads
        Worker threads, or ``None`` for every available core.  Must be ``>= 1``.

    Returns
    -------
    tuple of (indptr, indices, data)
        The item-item weight matrix ``W`` in **CSC** layout: segment ``i``,
        namely ``indices[indptr[i]:indptr[i + 1]]``, holds the neighbours ``k``
        of target item ``i`` in ascending order, with strictly positive weights
        ``W[k, i]``.  Hence ``scores = X @ W``.

    Raises
    ------
    ValueError
        Malformed CSR structure, negative or non-finite ``data``, or an
        out-of-range parameter (including ``n_threads == 0``).
    TypeError
        An array argument has an unsupported dtype.
    """
