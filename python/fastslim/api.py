from __future__ import annotations

import warnings
from collections.abc import Iterator

import numpy as np
from scipy import sparse

from .native import solve_slim
from .validation import (
    Matrix,
    MatrixLike,
    check_integer,
    check_interaction_matrix,
    check_params,
)

__all__ = ["ConvergenceWarning", "fit", "predict", "recommend"]


def solve(
    matrix: sparse.csr_matrix,
    lambd: float,
    beta: float,
    max_iter: int,
    tol: float,
    n_threads: int | None,
) -> tuple[np.ndarray, ...]:
    """Call the Rust solver, the one place that touches the extension module.

    Returns ``(indptr, indices, data, n_passes, converged)``, with the first
    three describing ``W`` in CSC layout.
    """
    return solve_slim(
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        n_rows=matrix.shape[0],
        n_cols=matrix.shape[1],
        lambd=lambd,
        beta=beta,
        max_iter=max_iter,
        tol=tol,
        n_threads=n_threads,
    )


class ConvergenceWarning(UserWarning):
    """Some items exhausted ``max_iter`` before every weight change fell below ``tol``.

    Their weights are the feasible iterate at cut-off, not the exact optimum.
    """


def fit_with_diagnostics(
    interaction_matrix: MatrixLike,
    lambd: float,
    beta: float,
    max_iter: int,
    tol: float,
    n_threads: int | None,
) -> tuple[sparse.csr_matrix | sparse.csr_array, np.ndarray, np.ndarray]:
    """Validate, solve and assemble ``W``; shared by :func:`fit` and ``SLIM``.

    Returns ``(weights, n_passes, converged)``.
    """
    lambd, beta, max_iter, tol, n_threads = check_params(
        lambd, beta, max_iter, tol, n_threads
    )
    matrix, is_sparse_array = check_interaction_matrix(interaction_matrix)
    n_items = matrix.shape[1]

    indptr, indices, data, n_passes, converged = solve(
        matrix, lambd, beta, max_iter, tol, n_threads
    )

    if not converged.all():
        n_bad = int(np.count_nonzero(~converged))
        warnings.warn(
            f"{n_bad} of {n_items} items did not converge within "
            f"max_iter={max_iter} passes (tol={tol}); the returned weights are "
            "a truncated solution. Increase max_iter, or increase beta to "
            "improve conditioning. SLIM(...).fit(X).converged reports which "
            "items are affected; silence this with "
            "warnings.filterwarnings('ignore', category=fastslim.ConvergenceWarning).",
            ConvergenceWarning,
            stacklevel=3,
        )

    # The solver emits W column by column, which is exactly CSC.
    csc_container = sparse.csc_array if is_sparse_array else sparse.csc_matrix
    weights = csc_container((data, indices, indptr), shape=(n_items, n_items)).tocsr()
    return weights, n_passes, converged


def fit(
    interaction_matrix: MatrixLike,
    lambd: float = 0.5,
    beta: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-4,
    n_threads: int | None = None,
) -> sparse.csr_matrix | sparse.csr_array:
    """Fit SLIM item-item weights with non-negative coordinate descent.

    Returns ``W`` of shape ``(n_items, n_items)``, sparse and non-negative with
    a zero diagonal, in a container that mirrors the input; see `docs/api.md`.
    """
    weights, _, _ = fit_with_diagnostics(
        interaction_matrix, lambd, beta, max_iter, tol, n_threads
    )
    return weights


def as_weights(weights: MatrixLike) -> Matrix:
    """Return ``weights`` as a CSR matrix or a 2-D dense float64 array."""
    if sparse.issparse(weights):
        if weights.ndim != 2:
            raise ValueError(
                f"weights must be 2-D (items x items), got {weights.ndim}-D"
            )
        return weights if weights.format == "csr" else weights.tocsr()
    dense = np.asarray(weights, dtype=np.float64)
    if dense.ndim != 2:
        raise ValueError(f"weights must be 2-D (items x items), got {dense.ndim}-D")
    return dense


def prepare_history(user_history: MatrixLike, n_items: int) -> tuple[Matrix, bool]:
    """Return ``(history, is_single_user)`` with ``history`` always 2-D.

    A sparse *matrix* of shape ``(1, n_items)`` counts as one user because
    ``spmatrix`` cannot be 1-D; a ``csr_array`` of that shape stays a batch.
    """
    if sparse.issparse(user_history):
        single = user_history.ndim == 1 or (
            not isinstance(user_history, sparse.sparray) and user_history.shape[0] == 1
        )
        history = user_history
        if history.ndim == 1:
            history = history.reshape(1, -1)
        if history.format != "csr":
            history = history.tocsr()
        if history.dtype != np.float64:
            history = history.astype(np.float64)
    else:
        dense = np.asarray(user_history)
        if dense.ndim == 1:
            single, dense = True, dense[np.newaxis, :]
        elif dense.ndim == 2:
            single = False
        else:
            raise ValueError(
                f"user_history must be 1-D or 2-D, got a {dense.ndim}-D array "
                f"with shape {dense.shape}"
            )
        history = dense.astype(np.float64, copy=False)

    if history.shape[1] != n_items:
        raise ValueError(
            f"user_history has {history.shape[1]} items but weights describe {n_items}"
        )
    return history, single


def mask_seen(scores: np.ndarray, history: Matrix) -> None:
    """Set the score of every nonzero history entry to ``-inf``, in place."""
    if isinstance(history, np.ndarray):
        scores[history != 0] = -np.inf
        return
    if history.nnz == 0:
        return
    rows = np.repeat(np.arange(scores.shape[0]), np.diff(history.indptr))
    nonzero = history.data != 0
    scores[rows[nonzero], history.indices[nonzero]] = -np.inf


def score_chunk(history: Matrix, weights: Matrix, exclude_seen: bool) -> np.ndarray:
    """Dense ``float64`` scores for one slice of users."""
    product = history @ weights
    scores = product.toarray() if sparse.issparse(product) else np.asarray(product)
    scores = scores.astype(np.float64, copy=False)
    if exclude_seen:
        mask_seen(scores, history)
    return scores


def chunks(n_rows: int, batch_size: int | None) -> Iterator[tuple[int, int]]:
    if batch_size is None:
        yield 0, n_rows
        return
    batch_size = check_integer(batch_size, "batch_size", 1)
    for start in range(0, n_rows, batch_size):
        yield start, min(start + batch_size, n_rows)


def scored_chunks(
    history: Matrix, weights: Matrix, exclude_seen: bool, batch_size: int | None
) -> Iterator[tuple[slice, np.ndarray]]:
    """Yield ``(rows, scores)`` for consecutive slices of users."""
    for start, stop in chunks(np.shape(history)[0], batch_size):
        rows = slice(start, stop)
        # pyrefly: ignore[bad-argument-type]  # scipy is untyped; slices infer as sparray
        yield rows, score_chunk(history[rows], weights, exclude_seen)


def top_k_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the ``k`` highest scores per row, best first.

    Ties among the selected items break by ascending item index; which items
    are selected on a tie at rank ``k`` is left to ``argpartition``.
    """
    n_rows, n_items = scores.shape
    if k == 0:
        return np.empty((n_rows, 0), dtype=np.int64)
    if k >= n_items:
        candidates = np.broadcast_to(np.arange(n_items), scores.shape)
        candidate_scores = scores
    else:
        candidates = np.argpartition(scores, n_items - k, axis=1)[:, n_items - k :]
        candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    # lexsort's *last* key is primary: descending score, then ascending index.
    order = np.lexsort((candidates, -candidate_scores), axis=-1)
    return np.take_along_axis(candidates, order, axis=1).astype(np.int64, copy=False)


def predict(
    weights: MatrixLike,
    user_history: MatrixLike,
    *,
    exclude_seen: bool = True,
    batch_size: int | None = None,
) -> np.ndarray:
    """Score every item for one user or a batch of users, as ``history @ weights``.

    Returns ``float64`` scores of shape ``(n_items,)`` or ``(n_users, n_items)``;
    see `docs/api.md`.
    """
    weight_matrix = as_weights(weights)
    n_sources, n_items = np.shape(weight_matrix)
    history, single = prepare_history(user_history, n_sources)
    scores = np.empty((np.shape(history)[0], n_items), dtype=np.float64)
    for rows, chunk in scored_chunks(history, weight_matrix, exclude_seen, batch_size):
        scores[rows] = chunk
    return scores[0] if single else scores


def recommend(
    weights: MatrixLike,
    user_history: MatrixLike,
    k: int = 10,
    *,
    exclude_seen: bool = True,
    batch_size: int | None = None,
) -> np.ndarray:
    """Recommend the top ``k`` items for one user or a batch of users.

    Returns ``int64`` item ids ranked best first, of shape ``(k,)`` or
    ``(n_users, k)``; see `docs/api.md`.
    """
    weight_matrix = as_weights(weights)
    n_sources, n_items = np.shape(weight_matrix)
    history, single = prepare_history(user_history, n_sources)
    k = min(check_integer(k, "k", 1), n_items)
    top = np.empty((np.shape(history)[0], k), dtype=np.int64)
    for rows, scores in scored_chunks(history, weight_matrix, exclude_seen, batch_size):
        top[rows] = top_k_from_scores(scores, k)
    return top[0] if single else top
