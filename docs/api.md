# API reference

Everything in `fastslim.__all__`, plus the metrics module and the compiled extension.
The algorithm behind `fit` is derived in [`algorithm.md`](algorithm.md).

## `fastslim.fit`

```python
fit(interaction_matrix, lambd=0.5, beta=0.5, max_iter=1000, tol=1e-4, n_threads=None)
```

Fit SLIM item-item weights with non-negative coordinate descent. For every item `i` the
solver minimises

```text
0.5 * ||x_i - X w||^2 + lambd * sum_k w_k + (beta / 2) * sum_k w_k^2
subject to  w >= 0,  w_i = 0
```

where `X` is the user-item matrix and `x_i` its `i`-th column. Non-negativity turns the
L1 term into a plain linear penalty, so the problem is a non-negative elastic net whose
solution is unique whenever `beta > 0`.

**Parameters**

| Name | Type | Meaning |
| --- | --- | --- |
| `interaction_matrix` | sparse matrix, sparse array or array-like | User-item interactions, shape `(n_users, n_items)`. Any SciPy sparse format (`csr`, `csc`, `coo`, `lil`, `dok`, ...), a sparse array, a dense `ndarray` or a nested sequence is accepted and converted to canonical CSR `float64`. Values must be finite and non-negative; explicit zeros are dropped. The input is not modified. |
| `lambd` | float, default `0.5` | L1 penalty, controls sparsity. It is compared against entries of the Gram matrix `P = X.T @ X`: for binary data those are raw co-occurrence counts, so a useful `lambd` scales with how often items co-occur (single digits for MovieLens-sized data, not `1e-4`). A neighbour `k` of item `i` can only enter the model when `P_ik >= lambd`, so raising `lambd` prunes weights outright. |
| `beta` | float, default `0.5` | L2 penalty, shrinks weights towards zero without pruning them. It also regularises the per-coordinate denominator `P_kk + beta`, which keeps very popular items from dominating and makes the solution unique. On count-scale data `P_kk` can reach the thousands while `beta` is a fraction; raising `beta` improves the conditioning and so the number of passes a fit needs. |
| `max_iter` | int, default `1000` | Maximum number of coordinate-descent passes per item. Full passes over every candidate and passes over the active set both count towards it. `0` returns an all-zero weight matrix. |
| `tol` | float, default `1e-4` | Convergence tolerance on the largest weight change within a pass. An item is done once a full pass moves no weight by `tol` or more, so `tol=0` means exactly `max_iter` passes. |
| `n_threads` | int or None, default `None` | Worker threads. `None` uses every available core. The result is bit-for-bit identical regardless of this value. |

**Returns** — `scipy.sparse.csr_matrix` or `scipy.sparse.csr_array`

Item-item weight matrix `W` of shape `(n_items, n_items)` with a zero diagonal and
strictly positive stored values, in canonical CSR form. Scores for a batch of users are
`user_history @ W`. The container mirrors the input: a SciPy *sparse array* in gives a
`csr_array` out, anything else gives a `csr_matrix`.

**Raises**

- `ValueError` — the input is not 2-D, holds negative or non-finite values, or a
  hyperparameter is out of range.
- `TypeError` — a hyperparameter has the wrong type (for example `max_iter=1.5`).

**Warns**

- `ConvergenceWarning` — some items exhausted `max_iter` before a full pass came back
  under `tol`. Their columns of `W` are a truncated solution: still a usable model, but
  not the optimum. Raise `max_iter`, or raise `beta` to improve the conditioning.

## `fastslim.predict`

```python
predict(weights, user_history, *, exclude_seen=True, batch_size=None)
```

Score every item for one user or a batch of users. Scores are `user_history @ weights`.

**Parameters**

| Name | Type | Meaning |
| --- | --- | --- |
| `weights` | sparse matrix, sparse array or ndarray | Item-item weight matrix, shape `(n_items, n_items)`, as returned by `fit`. |
| `user_history` | array-like or sparse | Either one user's interactions — a 1-D dense vector, a 1-D sparse array or a sparse *matrix* of shape `(1, n_items)` — or a batch of shape `(n_users, n_items)`. |
| `exclude_seen` | bool, default `True` | Replace the score of every item the user has already interacted with (any nonzero history entry) by `-inf`. |
| `batch_size` | int or None, default `None` | Score this many users at a time. Bounds the size of the intermediate sparse product for large batches; the returned array is dense either way. `None` scores everything in one go. |

**Returns** — `numpy.ndarray`

`float64` scores of shape `(n_items,)` for a single history, or `(n_users, n_items)` for
a batch.

**Raises** — `ValueError` if `user_history` is neither 1-D nor 2-D, or its width does not
match `weights`.

## `fastslim.recommend`

```python
recommend(weights, user_history, k=10, *, exclude_seen=True, batch_size=None)
```

Recommend the top `k` items for one user or a batch of users.

**Parameters**

| Name | Type | Meaning |
| --- | --- | --- |
| `weights` | sparse matrix, sparse array or ndarray | Item-item weight matrix, shape `(n_items, n_items)`. |
| `user_history` | array-like or sparse | One user's interactions or a batch; see `predict`. |
| `k` | int, default `10` | Number of items to return, clipped to `n_items`. |
| `exclude_seen` | bool, default `True` | Drop items the user has already interacted with. If fewer than `k` unseen items exist, seen items fill the remaining slots. |
| `batch_size` | int or None, default `None` | Recommend for this many users at a time. Unlike `predict` this genuinely bounds peak memory, since only `k` items per user are kept. |

**Returns** — `numpy.ndarray`

`int64` item indices sorted by descending score, of shape `(k,)` for a single history or
`(n_users, k)` for a batch. Ties among the selected items are broken by ascending item
index. Which items are selected when the score at the `k`-th position is tied is left to
`argpartition`: deterministic for a given input, but not necessarily index-ordered.

**Raises** — `ValueError` or `TypeError` for a bad `k` or `batch_size`, plus everything
`predict` raises.

## `fastslim.SLIM`

```python
SLIM(lambd=0.5, beta=0.5, max_iter=1000, tol=1e-4, n_threads=None)
```

A scikit-learn-flavoured estimator over the same solver, with no scikit-learn
dependency. The constructor stores hyperparameters untouched and validation happens in
`fit`.

**Parameters** — identical to `fit`'s `lambd`, `beta`, `max_iter`, `tol` and
`n_threads`.

**Attributes set by `fit`**

| Name | Type | Meaning |
| --- | --- | --- |
| `weights` | `scipy.sparse.csr_matrix` or `csr_array` | Item-item weights, shape `(n_items, n_items)`. |
| `n_items` | int | Number of items seen during `fit`. |
| `n_passes` | `numpy.ndarray` | `int64`, shape `(n_items,)`: coordinate-descent passes actually used for each item (full and active-set passes both count). |
| `converged` | `numpy.ndarray` | `bool`, shape `(n_items,)`: whether each item reached `tol` before running out of `max_iter`. Items with no candidate neighbours count as converged in zero passes. |

**Methods**

| Signature | Meaning |
| --- | --- |
| `fit(X, y=None)` | Fit the weights on a user-item interaction matrix and return `self`. `y` is ignored; SLIM is unsupervised. Warns `ConvergenceWarning` exactly as `fastslim.fit` does. |
| `predict(X_history, *, exclude_seen=True, batch_size=None)` | As `fastslim.predict`, using the fitted weights. |
| `recommend(X_history, k=10, *, exclude_seen=True, batch_size=None)` | As `fastslim.recommend`, using the fitted weights. |
| `get_params(deep=True)` | The hyperparameters as a dict. `deep` is accepted for scikit-learn compatibility and makes no difference. |
| `set_params(**params)` | Set hyperparameters and return `self`. Raises `ValueError` on an unknown name. |
| `__repr__()` | Shows only the hyperparameters that differ from the defaults; the result is valid Python that rebuilds an equivalent unfitted estimator. |

Calling `predict` or `recommend` before `fit` raises `fastslim.NotFittedError`.

## `fastslim.metrics`

```python
precision_at_k(predictions, test_matrix, k=10)
recall_at_k(predictions, test_matrix, k=10)
ndcg_at_k(predictions, test_matrix, k=10)
```

Every metric takes the model's output for a set of users and a sparse matrix of held-out
interactions, and averages over the users that actually have held-out items. Users with
an empty test row are skipped rather than scored as zero — averaging them in would make
the numbers depend on how many users the split happened to leave empty.

`predictions` may be given two ways, and **the dtype decides which reading applies**:

- a **score** array of shape `(n_users, n_items)` with a floating dtype, as returned by
  `predict`; the top `k` is taken here, or
- a **recommendation** array of shape `(n_users, >= k)` with an integer dtype, as
  returned by `recommend`, already ranked best-first.

| Metric | Value |
| --- | --- |
| `precision_at_k` | Fraction of the top `k` recommendations that are held-out items. The denominator is `k` even when a user has fewer than `k` held-out items. |
| `recall_at_k` | Fraction of a user's held-out items that appear in the top `k`. Users with more than `k` held-out items cannot reach 1.0. |
| `ndcg_at_k` | Normalised discounted cumulative gain with binary relevance: a hit at rank `j` (0-based) contributes `1 / log2(j + 2)`. The ideal DCG puts `min(n_held_out, k)` hits at the top ranks, so a user with fewer than `k` held-out items can still score 1.0. |

Each returns a `float`: the mean over users with at least one held-out item, or `0.0` if
there are none. `test_matrix` is a sparse matrix or array-like of shape
`(n_users, n_items)`. A `predictions` array that is not 2-D, does not cover the same
number of users as `test_matrix`, or (for scores) does not score the same number of
items, raises `ValueError`; so does an integer `predictions` array with fewer than `k`
ranked items per user.

## `fastslim.ConvergenceWarning`

A `UserWarning` subclass, emitted by `fit` and `SLIM.fit` when some items exhausted
`max_iter` before every weight change fell below `tol`. Their weights are the feasible
iterate at cut-off rather than the exact optimum. Raise `max_iter`, or raise `beta` to
improve conditioning; `SLIM(...).fit(X).converged` tells which items are affected.
Silence it with
`warnings.filterwarnings("ignore", category=fastslim.ConvergenceWarning)`.

## `fastslim.NotFittedError`

Raised when a `SLIM` instance is used before `fit`. It inherits from both `ValueError`
and `AttributeError`, so `except ValueError` and `except AttributeError` both catch it,
matching scikit-learn's exception of the same name.

## `fastslim.__version__`

The installed version string, single-sourced from `Cargo.toml` through maturin.

## `fastslim.native.solve_slim`

The compiled extension. It is not part of the supported surface — call `fastslim.fit` —
but the tests and the type stub (`python/fastslim/native.pyi`) use it directly.

```python
solve_slim(data, indices, indptr, n_rows, n_cols,
           lambd=0.5, beta=0.5, max_iter=1000, tol=1e-4, n_threads=None)
```

`data`, `indices` and `indptr` are the CSR arrays of the `(n_rows, n_cols)` user-item
matrix. `data` must be `float64`, finite and non-negative; the index arrays may be
`int32` or `int64`, and `indptr` has `n_rows + 1` entries. Duplicate column entries
within a row are summed and indices need not be sorted.

Returns `(indptr, indices, data, n_passes, converged)`:

- `indptr` (`int64`, `n_cols + 1`), `indices` (`int64`, nnz) and `data` (`float64`, nnz)
  are `W` in **CSC** layout: segment `i`, namely `indices[indptr[i]:indptr[i + 1]]`,
  holds the neighbours `k` of target item `i` in ascending order with strictly positive
  weights `W[k, i]`. Hence `scores = X @ W`.
- `n_passes` (`int64`, `n_cols`) counts the passes used per item, active-set and full
  passes alike.
- `converged` (`bool`, `n_cols`) says whether each item met `tol` within `max_iter`. A
  converged item satisfies the KKT conditions to within `(P_kk + beta) * tol` per
  coordinate. `False` means the item was cut off (always the case for `max_iter = 0` or
  `tol = 0`) and its weights are a truncated, though feasible, solution.

Raises `ValueError` on a malformed CSR structure, negative or non-finite `data`, an
out-of-range parameter (including `n_threads == 0`), or interaction values so large that
the Gram matrix `X.T @ X` overflows; `TypeError` for an unsupported array dtype.

`src/lib.rs` is the authority on this tuple; this section and the stub mirror it.
