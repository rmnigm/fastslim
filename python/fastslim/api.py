"""Functional API: :func:`fit`, :func:`predict` and :func:`recommend`."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any

import numpy as np
from scipy import sparse

from .native import solve_slim
from .validation import check_integer, check_interaction_matrix, check_params

__all__ = ["ConvergenceWarning", "fit", "predict", "recommend"]

SPARRAY: Any = getattr(sparse, "sparray", ())


def solve(
    matrix: Any,
    lambd: float,
    beta: float,
    max_iter: int,
    tol: float,
    n_threads: int | None,
) -> tuple[np.ndarray, ...]:
    """Call the Rust solver on a canonical CSR matrix.

    The single point of contact with the extension module.  Everything the
    solver returns passes through here, so a change to the binding's return
    tuple only has to be absorbed in this function and its callers.  The shape
    of that tuple is documented in ``src/lib.rs``: it is
    ``(indptr, indices, data, n_passes, converged)``, where the first three
    arrays describe ``W`` in CSC layout and the last two are per-item solver
    diagnostics (``int64`` passes used, ``bool`` converged flag).
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

    Their weights are the feasible iterate at cut-off rather than the exact
    optimum.  Raise ``max_iter``, or raise ``beta`` to improve conditioning;
    ``SLIM(...).fit(X).converged`` tells which items are affected.
    """


def fit_with_diagnostics(
    interaction_matrix: Any,
    lambd: float,
    beta: float,
    max_iter: int,
    tol: float,
    n_threads: int | None,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Validate, solve and assemble ``W``; shared by :func:`fit` and ``SLIM``.

    Returns ``(weights, n_passes, converged)``.  Emits
    :class:`ConvergenceWarning` when any item was truncated; ``stacklevel`` is
    chosen so the warning points at the code that called ``fit``/``SLIM.fit``.
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

    # The solver returns W column by column (one segment per target item),
    # which is exactly CSC; converting gives canonical, sorted CSR.
    csc_container = sparse.csc_array if is_sparse_array else sparse.csc_matrix
    weights = csc_container((data, indices, indptr), shape=(n_items, n_items)).tocsr()
    return weights, n_passes, converged


def fit(
    interaction_matrix: Any,
    lambd: float = 0.5,
    beta: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-4,
    n_threads: int | None = None,
) -> sparse.csr_matrix:
    r"""Fit SLIM item-item weights with non-negative coordinate descent.

    For every item :math:`i` the solver minimises

    .. math::

        \tfrac{1}{2} \lVert x_i - X w \rVert_2^2
        + \lambda \sum_k w_k
        + \tfrac{\beta}{2} \sum_k w_k^2 ,
        \qquad w \ge 0, \; w_i = 0 ,

    where :math:`X` is the user-item matrix and :math:`x_i` its :math:`i`-th
    column.  The non-negativity constraint turns the :math:`\ell_1` term into a
    plain linear penalty, so the problem is a non-negative elastic net and the
    solution is unique whenever ``beta > 0``.

    Parameters
    ----------
    interaction_matrix : sparse matrix, sparse array or array-like
        User-item interactions, shape ``(n_users, n_items)``.  Any scipy sparse
        format (``csr``, ``csc``, ``coo``, ``lil``, ``dok``, ...), a scipy
        sparse array, a dense ``ndarray`` or a nested sequence is accepted and
        converted to canonical CSR ``float64``.  Values must be finite and
        non-negative; explicit zeros are dropped.  The input is not modified.
    lambd : float, default=0.5
        L1 penalty, controls sparsity.  It is compared against entries of the
        Gram matrix :math:`P = X^\top X`: for binary data those are raw
        co-occurrence counts, so a useful ``lambd`` scales with how often items
        co-occur (single digits for MovieLens-sized data, not ``1e-4``).  A
        neighbour :math:`k` of item :math:`i` can only enter the model when
        :math:`P_{ik} \ge \lambda`, so raising ``lambd`` prunes weights outright.
    beta : float, default=0.5
        L2 penalty, shrinks weights towards zero without pruning them.  It also
        regularises the per-coordinate denominator :math:`P_{kk} + \beta`, which
        keeps very popular items from dominating and makes the solution unique.
        On count-scale data :math:`P_{kk}` can reach the thousands while
        ``beta`` is a fraction; raising ``beta`` improves the conditioning and
        so the number of passes a fit needs.
    max_iter : int, default=1000
        Maximum number of coordinate-descent passes per item.  Full passes over
        every candidate and passes over the active set both count towards it.
        ``0`` returns an all-zero weight matrix.
    tol : float, default=1e-4
        Convergence tolerance on the largest weight change within a pass.  An
        item is done once a full pass moves no weight by ``tol`` or more, so
        ``tol=0`` means exactly ``max_iter`` passes.
    n_threads : int or None, default=None
        Worker threads.  ``None`` uses every available core.  The result is
        bit-for-bit identical regardless of this value.

    Returns
    -------
    scipy.sparse.csr_matrix or scipy.sparse.csr_array
        Item-item weight matrix ``W`` of shape ``(n_items, n_items)`` with a
        zero diagonal and strictly positive stored values, in canonical CSR
        form.  Scores for a batch of users are ``user_history @ W``.  The
        container mirrors the input: a scipy *sparse array* in gives a
        ``csr_array`` out, anything else gives a ``csr_matrix``.

    Raises
    ------
    ValueError
        If the input is not 2-D, holds negative or non-finite values, or a
        hyperparameter is out of range.
    TypeError
        If a hyperparameter has the wrong type (e.g. ``max_iter=1.5``).

    Warns
    -----
    ConvergenceWarning
        If some items exhausted ``max_iter`` before a full pass came back
        under ``tol``.  Their columns of ``W`` are a truncated solution:
        still a usable model, but not the optimum.  Raise ``max_iter``, or
        raise ``beta`` to improve the conditioning.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse
    >>> import fastslim
    >>> X = sparse.csr_matrix(np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]]))
    >>> W = fastslim.fit(X, lambd=0.1, beta=0.1)
    >>> W.shape
    (3, 3)
    >>> bool(np.all(W.diagonal() == 0))
    True
    """
    weights, _, _ = fit_with_diagnostics(
        interaction_matrix, lambd, beta, max_iter, tol, n_threads
    )
    return weights


def as_weights(weights: Any) -> Any:
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


def prepare_history(user_history: Any, n_items: int) -> tuple[Any, bool]:
    """Return ``(history, is_single_user)`` with ``history`` always 2-D.

    A 1-D input -- dense or a 1-D sparse array -- describes one user, and so
    does a sparse *matrix* of shape ``(1, n_items)``: ``spmatrix`` cannot be
    1-D, so that is the only way to spell a single history with one.  A
    ``csr_array`` of shape ``(1, n_items)`` stays a one-user batch, because
    with sparse arrays 1-D is expressible.
    """
    if sparse.issparse(user_history):
        single = user_history.ndim == 1 or (
            not isinstance(user_history, SPARRAY) and user_history.shape[0] == 1
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


def mask_seen(scores: np.ndarray, history: Any) -> None:
    """Set the score of every nonzero history entry to ``-inf``, in place."""
    if sparse.issparse(history):
        if history.nnz == 0:
            return
        rows = np.repeat(np.arange(history.shape[0]), np.diff(history.indptr))
        nonzero = history.data != 0
        scores[rows[nonzero], history.indices[nonzero]] = -np.inf
    else:
        scores[history != 0] = -np.inf


def score_chunk(history: Any, weights: Any, exclude_seen: bool) -> np.ndarray:
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


def top_k_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` highest scores per row, best first.

    Ties among the selected items are broken by ascending item index, so the
    ordering is reproducible.  Which items are selected when the score at the
    ``k``-th position is tied is left to ``argpartition``; it is deterministic
    for a given input but not necessarily index-ordered.
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
    weights: Any,
    user_history: Any,
    *,
    exclude_seen: bool = True,
    batch_size: int | None = None,
) -> np.ndarray:
    """Score every item for one user or a batch of users.

    Scores are ``user_history @ weights``.

    Parameters
    ----------
    weights : sparse matrix, sparse array or ndarray
        Item-item weight matrix, shape ``(n_items, n_items)``, as returned by
        :func:`fit`.
    user_history : array-like or sparse
        Either one user's interactions -- a 1-D dense vector, a 1-D sparse
        array or a sparse *matrix* of shape ``(1, n_items)`` -- or a batch of
        shape ``(n_users, n_items)``.
    exclude_seen : bool, default=True
        Replace the score of every item the user has already interacted with
        (any nonzero history entry) by ``-inf``.
    batch_size : int or None, default=None
        Score this many users at a time.  Bounds the size of the intermediate
        sparse product for large batches; the returned array is dense either
        way.  ``None`` scores everything in one go.

    Returns
    -------
    numpy.ndarray
        ``float64`` scores of shape ``(n_items,)`` for a single history, or
        ``(n_users, n_items)`` for a batch.
    """
    weight_matrix = as_weights(weights)
    history, single = prepare_history(user_history, weight_matrix.shape[0])
    n_users = history.shape[0]
    scores = np.empty((n_users, weight_matrix.shape[1]), dtype=np.float64)
    for start, stop in chunks(n_users, batch_size):
        scores[start:stop] = score_chunk(
            history[start:stop], weight_matrix, exclude_seen
        )
    return scores[0] if single else scores


def recommend(
    weights: Any,
    user_history: Any,
    k: int = 10,
    *,
    exclude_seen: bool = True,
    batch_size: int | None = None,
) -> np.ndarray:
    """Recommend the top ``k`` items for one user or a batch of users.

    Parameters
    ----------
    weights : sparse matrix, sparse array or ndarray
        Item-item weight matrix, shape ``(n_items, n_items)``.
    user_history : array-like or sparse
        One user's interactions or a batch; see :func:`predict`.
    k : int, default=10
        Number of items to return, clipped to ``n_items``.
    exclude_seen : bool, default=True
        Drop items the user has already interacted with.  If fewer than ``k``
        unseen items exist, seen items fill the remaining slots.
    batch_size : int or None, default=None
        Recommend for this many users at a time.  Unlike :func:`predict` this
        genuinely bounds peak memory, since only ``k`` items per user are kept.

    Returns
    -------
    numpy.ndarray
        ``int64`` item indices sorted by descending score, of shape ``(k,)``
        for a single history or ``(n_users, k)`` for a batch.
    """
    weight_matrix = as_weights(weights)
    history, single = prepare_history(user_history, weight_matrix.shape[0])
    n_items = weight_matrix.shape[1]
    k = min(check_integer(k, "k", 1), n_items)
    n_users = history.shape[0]
    top = np.empty((n_users, k), dtype=np.int64)
    for start, stop in chunks(n_users, batch_size):
        scores = score_chunk(history[start:stop], weight_matrix, exclude_seen)
        top[start:stop] = top_k_from_scores(scores, k)
    return top[0] if single else top
