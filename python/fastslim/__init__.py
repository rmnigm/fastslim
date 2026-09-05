"""SLIM (Sparse Linear Methods) solver, implemented in Rust."""

from typing import Optional, Union
import numpy as np
from scipy import sparse

from ._slim_rs import solve_slim as _solve_slim, SlimResult


def fit(
    interaction_matrix: sparse.spmatrix,
    lambd: float = 0.5,
    beta: float = 0.5,
    max_iter: int = 100,
    n_threads: Optional[int] = None,
) -> sparse.csr_matrix:
    """
    Fit SLIM model using coordinate descent.

    Optimizes the following loss for each item i:
    L_i = 0.5 * ||a_i - sum_j w_ij * a_j||^2 + lambda * sum_j |w_ij| + beta * sum_j w_ij^2

    Parameters
    ----------
    interaction_matrix : scipy.sparse.spmatrix
        User-item interaction matrix (users x items).
        Will be converted to CSR format if needed.
    lambd : float, default=0.5
        L1 regularization coefficient (controls sparsity).
    beta : float, default=0.5
        L2 regularization coefficient (controls magnitude).
    max_iter : int, default=100
        Maximum number of coordinate descent iterations per item.
    n_threads : int, optional
        Number of threads for parallel computation.
        If None, uses all available cores.

    Returns
    -------
    scipy.sparse.csr_matrix
        Item-item similarity matrix W (items x items).
        Predictions: scores = interaction_matrix @ W
    """
    if not sparse.isspmatrix_csr(interaction_matrix):
        interaction_matrix = sparse.csr_matrix(interaction_matrix)

    interaction_matrix = interaction_matrix.astype(np.float64)
    n_users, n_items = interaction_matrix.shape

    result: SlimResult = _solve_slim(
        data=interaction_matrix.data,
        indices=interaction_matrix.indices.astype(np.int64),
        indptr=interaction_matrix.indptr.astype(np.int64),
        n_rows=n_users,
        n_cols=n_items,
        lambd=lambd,
        beta=beta,
        max_iter=max_iter,
        n_threads=n_threads,
    )

    weight_matrix = sparse.coo_matrix(
        (result.data, (result.rows, result.cols)),
        shape=result.shape,
    ).tocsr()

    return weight_matrix


def predict(
    weights: sparse.spmatrix,
    user_history: Union[sparse.spmatrix, np.ndarray],
    *,
    exclude_seen: bool = True,
) -> np.ndarray:
    """
    Predict item scores for a single user given item-item weights.

    Parameters
    ----------
    weights : scipy.sparse.spmatrix
        Item-item weight matrix (items x items).
    user_history : scipy.sparse.spmatrix or numpy.ndarray
        User interaction vector (items,) or a 1 x items sparse row.
    exclude_seen : bool, default=True
        If True, set scores for seen items to -inf.

    Returns
    -------
    numpy.ndarray
        Dense score vector of shape (items,).
    """
    if not sparse.isspmatrix_csr(weights):
        weights = sparse.csr_matrix(weights)
    if sparse.issparse(user_history):
        history_csr = sparse.csr_matrix(user_history)
        scores = (history_csr @ weights).toarray().ravel()
        if exclude_seen:
            scores[history_csr.indices] = -np.inf
        return scores
    history = np.asarray(user_history)
    if history.ndim == 2:
        history = history.ravel()
    scores = np.asarray(history @ weights).ravel()
    if exclude_seen:
        scores[history > 0] = -np.inf
    return scores


__all__ = ["fit", "predict"]
__version__ = "0.1.3"
