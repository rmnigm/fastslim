import numpy as np
import numpy.typing as npt

Indices = npt.NDArray[np.int32] | npt.NDArray[np.int64]

def solve_slim(
    data: npt.NDArray[np.float64],
    indices: Indices,
    indptr: Indices,
    n_rows: int,
    n_cols: int,
    lambd: float = ...,
    beta: float = ...,
    max_iter: int = ...,
    tol: float = ...,
    n_threads: int | None = ...,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.bool_],
]:
    """Fit SLIM item-item weights on a CSR user-item matrix.

    Returns ``(indptr, indices, data, n_passes, converged)``: ``W`` in CSC
    layout plus per-item diagnostics.  See `docs/api.md` and `src/lib.rs`.
    """
