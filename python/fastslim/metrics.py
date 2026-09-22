from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import sparse

from .api import top_k_from_scores
from .validation import MatrixLike, check_integer, check_interaction_matrix

__all__ = ["ndcg_at_k", "precision_at_k", "recall_at_k"]


def as_test_csr(test_matrix: MatrixLike, n_users: int) -> sparse.csr_matrix:
    """Canonical CSR view of the held-out interactions."""
    test_csr, _ = check_interaction_matrix(test_matrix, "test_matrix")
    if test_csr.shape[0] != n_users:
        raise ValueError(
            f"test_matrix has {test_csr.shape[0]} rows but predictions cover "
            f"{n_users} users"
        )
    return test_csr


def check_ranked_ids(ranked: np.ndarray, n_items: int) -> None:
    """Reject item ids outside ``[0, n_items)`` and ids repeated within a row."""
    outside = (ranked < 0) | (ranked >= n_items)
    if outside.any():
        user, rank = np.argwhere(outside)[0]
        raise ValueError(
            f"predictions holds item id {ranked[user, rank]} for user {user}, "
            f"outside the {n_items} items of test_matrix"
        )
    ordered = np.sort(ranked, axis=1)
    repeated = ordered[:, 1:] == ordered[:, :-1]
    if repeated.any():
        user, rank = np.argwhere(repeated)[0]
        raise ValueError(
            f"predictions repeats item id {ordered[user, rank]} for user {user} "
            "within the top k"
        )


def top_k_items(predictions: np.ndarray, k: int, n_items: int) -> np.ndarray:
    """Coerce scores or recommendations to a ``(n_users, k)`` index array."""
    if np.issubdtype(predictions.dtype, np.integer):
        if predictions.shape[1] < k:
            raise ValueError(
                f"predictions holds only {predictions.shape[1]} ranked items "
                f"per user, need at least k={k}"
            )
        ranked = predictions[:, :k]
        check_ranked_ids(ranked, n_items)
        return ranked
    if predictions.shape[1] != n_items:
        raise ValueError(
            f"predictions score {predictions.shape[1]} items but test_matrix "
            f"has {n_items}"
        )
    return top_k_from_scores(predictions, min(k, n_items))


def hits(top_k: np.ndarray, test_csr: sparse.csr_matrix) -> np.ndarray:
    """Say whether ``top_k[u, j]`` is a held-out item for user ``u``.

    Folding the user into the key (``user * n_items + item``) turns the per-user
    membership test into one vectorised ``searchsorted``.
    """
    n_users, k = top_k.shape
    n_items = test_csr.shape[1]
    if k == 0 or test_csr.nnz == 0:
        return np.zeros((n_users, k), dtype=bool)

    rows = np.repeat(np.arange(n_users, dtype=np.int64), np.diff(test_csr.indptr))
    keys = rows * n_items + test_csr.indices.astype(np.int64)
    queries = (
        np.arange(n_users, dtype=np.int64)[:, np.newaxis] * n_items
        + top_k.astype(np.int64)
    ).ravel()

    position = np.searchsorted(keys, queries)
    inside = position < keys.size
    hit = np.zeros(queries.shape, dtype=bool)
    hit[inside] = keys[position[inside]] == queries[inside]
    return hit.reshape(n_users, k)


def prepare(
    predictions: npt.ArrayLike, test_matrix: MatrixLike, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(hits, n_test_per_user, has_test_items)`` for the given inputs."""
    k = check_integer(k, "k", 1)
    predictions = np.asarray(predictions)
    if predictions.ndim != 2:
        raise ValueError(
            f"predictions must be 2-D, got a {predictions.ndim}-D array with "
            f"shape {predictions.shape}"
        )
    test_csr = as_test_csr(test_matrix, predictions.shape[0])
    top_k = top_k_items(predictions, k, test_csr.shape[1])
    n_test = np.diff(test_csr.indptr)
    return hits(top_k, test_csr), n_test, n_test > 0


def precision_at_k(
    predictions: npt.ArrayLike, test_matrix: MatrixLike, k: int = 10
) -> float:
    """Fraction of the top ``k`` recommendations that are held-out items.

    Parameters
    ----------
    predictions : ndarray
        Scores ``(n_users, n_items)`` or ranked item ids ``(n_users, >= k)``.
    test_matrix : sparse matrix or array-like
        Held-out interactions, ``(n_users, n_items)``.
    k : int, default=10
        Cut-off.  The denominator is ``k`` even when a user has fewer than ``k``
        held-out items.

    Returns
    -------
    float
        Mean precision over users with at least one held-out item, or ``0.0``
        if there are none.
    """
    hits, _, has_test = prepare(predictions, test_matrix, k)
    if not has_test.any():
        return 0.0
    return float(np.mean(hits[has_test].sum(axis=1) / k))


def recall_at_k(
    predictions: npt.ArrayLike, test_matrix: MatrixLike, k: int = 10
) -> float:
    """Fraction of a user's held-out items that appear in the top ``k``.

    Parameters
    ----------
    predictions : ndarray
        Scores ``(n_users, n_items)`` or ranked item ids ``(n_users, >= k)``.
    test_matrix : sparse matrix or array-like
        Held-out interactions, ``(n_users, n_items)``.
    k : int, default=10
        Cut-off.  Users with more than ``k`` held-out items cannot reach 1.0.

    Returns
    -------
    float
        Mean recall over users with at least one held-out item, or ``0.0`` if
        there are none.
    """
    hits, n_test, has_test = prepare(predictions, test_matrix, k)
    if not has_test.any():
        return 0.0
    return float(np.mean(hits[has_test].sum(axis=1) / n_test[has_test]))


def ndcg_at_k(
    predictions: npt.ArrayLike, test_matrix: MatrixLike, k: int = 10
) -> float:
    """Normalised discounted cumulative gain at ``k``, with binary relevance.

    A hit at rank ``j`` (0-based) contributes ``1 / log2(j + 2)``.  The ideal
    DCG puts ``min(n_held_out, k)`` hits at the top ranks, so a user with fewer
    than ``k`` held-out items can still score 1.0.

    Parameters
    ----------
    predictions : ndarray
        Scores ``(n_users, n_items)`` or ranked item ids ``(n_users, >= k)``.
    test_matrix : sparse matrix or array-like
        Held-out interactions, ``(n_users, n_items)``.
    k : int, default=10
        Cut-off.

    Returns
    -------
    float
        Mean NDCG over users with at least one held-out item, or ``0.0`` if
        there are none.
    """
    hits, n_test, has_test = prepare(predictions, test_matrix, k)
    if not has_test.any():
        return 0.0
    discount = 1.0 / np.log2(np.arange(hits.shape[1]) + 2.0)
    dcg = hits[has_test] @ discount
    ideal = np.concatenate([[0.0], np.cumsum(discount)])
    idcg = ideal[np.minimum(n_test[has_test], hits.shape[1])]
    return float(np.mean(dcg / idcg))
