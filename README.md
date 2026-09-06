# fastslim

[![PyPI](https://img.shields.io/pypi/v/fastslim.svg)](https://pypi.org/project/fastslim/)
[![CI](https://img.shields.io/github/actions/workflow/status/rmnigm/fastslim/ci.yml?branch=main&label=CI)](https://github.com/rmnigm/fastslim/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/fastslim.svg)](https://pypi.org/project/fastslim/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/rmnigm/fastslim/blob/main/LICENSE)

SLIM ([Sparse Linear Methods](https://ieeexplore.ieee.org/document/6137254), Ning &
Karypis 2011) learns a sparse, non-negative item-item weight matrix `W` from a user-item
interaction matrix `X`, and scores users with a single sparse product `X @ W`. It is one
of the strongest top-N recommenders that stays fully interpretable — every
recommendation decomposes into "because you interacted with these items". `fastslim`
implements it in Rust: the solver is **exact** (it satisfies the KKT conditions of the
problem it documents, checked against an independent reference solver),
**deterministic** (bit-identical output for any thread count), **fast** (seconds where
0.1.x took minutes), and wrapped in a **tiny Python API** — `fit`, `predict`,
`recommend`, plus a scikit-learn-style estimator and three ranking metrics.

## Install

```bash
pip install fastslim
# or
uv add fastslim
```

Prebuilt wheels ship for Linux (x86_64, aarch64, manylinux), macOS (universal2) and
Windows (x64). They are `abi3` wheels, so one wheel per platform covers CPython 3.10 and
newer. The only runtime dependencies are NumPy and SciPy; a source distribution is
published too, and building it needs a Rust toolchain.

## Quickstart

### Fit and recommend

```python
import numpy as np
from scipy import sparse

import fastslim

rng = np.random.default_rng(0)
interactions = sparse.csr_matrix((rng.random((2000, 300)) < 0.03).astype(np.float64))

# Item-item weights: (n_items, n_items), sparse, non-negative, zero diagonal.
weights = fastslim.fit(interactions, lambd=2.0, beta=2.0, max_iter=50)
print(weights.shape, weights.nnz)

# Top-10 items for the first five users, excluding what they already have.
top = fastslim.recommend(weights, interactions[:5], k=10)
print(top.shape)

# Raw scores instead of ranks, for one user.
scores = fastslim.predict(weights, interactions[0])
print(scores.shape)
```

`fit` accepts any SciPy sparse matrix or array, a dense `ndarray`, or a nested sequence,
and converts it to canonical CSR `float64` internally without touching your input.
Values must be finite and non-negative. The output container mirrors the input: pass a
sparse *array* (`csr_array`, `coo_array`, ...) and you get a `csr_array` back, otherwise
a `csr_matrix`.

### The `SLIM` estimator

The same solver behind a scikit-learn-flavoured object (no scikit-learn dependency):

```python
from fastslim import SLIM

model = SLIM(lambd=2.0, beta=2.0, max_iter=50).fit(interactions)
print(model.n_items_, model.weights_.nnz)
print(model.recommend(interactions[0], k=5))
print(model.get_params())
```

Learned state lives in trailing-underscore attributes (`weights_`, `n_items_`,
`n_passes_`, `converged_`); using the model before `fit` raises
`fastslim.NotFittedError`, which subclasses both `ValueError` and `AttributeError`.

### Evaluating

`fastslim.metrics` takes either a score array from `predict` or a ranked-id array from
`recommend`, and a sparse matrix of held-out interactions. Users with no held-out items
are skipped rather than scored as zero.

```python
import numpy as np
from scipy import sparse

import fastslim
from fastslim import metrics

rng = np.random.default_rng(0)
dense = (rng.random((2000, 300)) < 0.03).astype(np.float64)
held_out = rng.random(dense.shape) < 0.2

train = sparse.csr_matrix(np.where(held_out, 0.0, dense))
test = sparse.csr_matrix(np.where(held_out, dense, 0.0))

weights = fastslim.fit(train, lambd=2.0, beta=2.0, max_iter=50)
top = fastslim.recommend(weights, train, k=10, batch_size=1000)

print(f"precision@10 {metrics.precision_at_k(top, test, k=10):.4f}")
print(f"recall@10    {metrics.recall_at_k(top, test, k=10):.4f}")
print(f"ndcg@10      {metrics.ndcg_at_k(top, test, k=10):.4f}")
```

The matrix above is uniform noise, so those scores only exercise the API; see
[Performance](#performance) for MovieLens numbers.

## API

| Object | What it does |
| --- | --- |
| `fastslim.fit(X, lambd=0.5, beta=0.5, max_iter=100, tol=1e-6, n_threads=None)` | Fit item-item weights `W`; returns a sparse `(n_items, n_items)` matrix with a zero diagonal and strictly positive stored values |
| `fastslim.predict(W, history, *, exclude_seen=True, batch_size=None)` | Dense `float64` scores (`history @ W`) for one user or a batch; seen items become `-inf` |
| `fastslim.recommend(W, history, k=10, *, exclude_seen=True, batch_size=None)` | `int64` item ids ranked best-first; `batch_size` bounds peak memory |
| `fastslim.SLIM(...)` | Estimator wrapper: `fit`, `predict`, `recommend`, `get_params`, `set_params`, `weights_`, `n_items_`, `n_passes_`, `converged_` |
| `fastslim.metrics.precision_at_k / recall_at_k / ndcg_at_k` | Top-`k` ranking metrics against a held-out sparse matrix |
| `fastslim.ConvergenceWarning` | `UserWarning` subclass emitted when some items exhaust `max_iter` |
| `fastslim.NotFittedError` | Raised by an unfitted `SLIM`; subclasses `ValueError` and `AttributeError` |
| `fastslim.__version__` | Installed version, single-sourced from the crate |

Inputs are validated up front: a wrong shape, a negative or non-finite value, or an
out-of-range hyperparameter raises `ValueError`; a hyperparameter of the wrong type
(`max_iter=1.5`, `max_iter=True`) raises `TypeError`. The package ships type stubs and
`py.typed`, so `W`'s layout and every signature are visible to type checkers.

## Choosing hyperparameters

**`lambd` — the L1 penalty, on the co-occurrence-count scale.** It is compared directly
against entries of the Gram matrix `P = X.T @ X`. For binary data `P_ik` is the raw
number of users who interacted with both `i` and `k`, so a useful `lambd` is a single
digit on MovieLens-sized data, not `1e-4`. A neighbour `k` can only enter item `i`'s
model when `P_ik >= lambd`, so raising `lambd` prunes candidates outright and shrinks
`W`. Sweep it on a log scale and watch `W.nnz` alongside your ranking metric; the
benchmarks below use `lambd=2.0`.

**`beta` — the L2 penalty, for conditioning and uniqueness.** It shrinks weights without
pruning them and enters every coordinate update through the denominator `P_kk + beta`,
which keeps very popular items from dominating. With `beta > 0` the per-item problem is
strictly convex, so the solution is unique; at `beta = 0` it need not be.

**`max_iter` and `tol` — the stopping rule.** `max_iter` caps the total number of
coordinate-descent passes per item (full passes and active-set passes both count), and
an item finishes when a full pass moves no weight by `tol` or more. `tol=0` therefore
means exactly `max_iter` passes, and `max_iter=0` returns an all-zero `W`.

When some items hit `max_iter` first, `fit` emits a `fastslim.ConvergenceWarning` naming
how many; `SLIM` additionally records `n_passes_` (passes used per item) and
`converged_` (a per-item boolean array). A truncated solution is still a valid model,
just not the exact optimum. The usual cause is conditioning: on count-scale data `P_kk`
runs into the thousands while `beta` is around `0.5`, which makes coordinate descent
crawl. Raise `beta`, or raise `max_iter`.

```python
import warnings

import fastslim

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    weights = fastslim.fit(interactions, lambd=2.0, beta=0.5, max_iter=3)

for entry in caught:
    print(f"{type(entry.message).__name__}: {entry.message}")
```

**`n_threads` — parallelism only, never the answer.** `None` uses every available core.
Items are solved independently and the Gram matrix is accumulated in a fixed summation
order, so the result is bit-for-bit identical for any thread count; threads change wall
clock and nothing else.

## Performance

### 0.1.x to 0.2.0

12 threads on Apple Silicon, median of 3 runs.

| Workload | Fit time 0.1.x | Fit time 0.2.0 | Peak RSS 0.1.x | Peak RSS 0.2.0 |
| --- | --- | --- | --- | --- |
| Synthetic 20,000 x 3,000, 1% density, `max_iter=20` | 65.3 s | 7.0 s | 1455 MiB | 757 MiB |
| MovieLens 1M (6,041 x 3,953, 1.0M nnz), `max_iter=15` | 25.6 s | 0.85 s | 3902 MiB | 400 MiB |

Single-threaded 0.2.0 on the synthetic problem takes 51.6 s, so most of the gain is
algorithmic (sparse Gram accumulation, residual maintenance, a correct active set)
rather than parallelism.

### Against implicit's ALS

Produced by
[`benchmarks/benchmark.py`](https://github.com/rmnigm/fastslim/blob/main/benchmarks/benchmark.py),
an 80/20 split at `seed=42`, `k=10`:

```bash
uv sync --group bench
uv run python benchmarks/benchmark.py --dataset 100k --baseline als --markdown
uv run python benchmarks/benchmark.py --dataset 1m   --baseline als --markdown
```

SLIM runs with the script's defaults (`--lambd 2.0 --beta 2.0 --max-iter 50 --seed 42`),
ALS with 64 latent factors.

**MovieLens 100k** — 944 users x 1,683 items, 100k interactions:

| Metric | fastslim SLIM | implicit ALS |
| --- | --- | --- |
| Fit time | 0.17 s | 0.58 s |
| precision@10 | **0.3391** | 0.2878 |
| recall@10 | **0.2236** | 0.2017 |
| ndcg@10 | **0.4069** | 0.3433 |
| Model size | 42,116 nonzeros | 168,128 factors |

**MovieLens 1M** — 6,041 users x 3,953 items, 1.0M interactions:

| Metric | fastslim SLIM | implicit ALS |
| --- | --- | --- |
| Fit time | 2.55 s | 4.45 s |
| precision@10 | **0.3620** | 0.3465 |
| recall@10 | **0.1652** | 0.1621 |
| ndcg@10 | **0.4100** | 0.3915 |
| Model size | 210,095 nonzeros | 639,616 factors |

The script also prints a peak-RSS column. That number is a process-wide high-water mark
taken across sequential fits, not an isolated per-model measurement, so it is left out
of the tables above.

## Algorithm

For every item `i`, independently, the solver minimises

```text
0.5 * ||x_i - X w||^2 + lambd * sum_k w_k + (beta / 2) * sum_k w_k^2
subject to  w >= 0,  w_i = 0
```

where `X` is the user-item matrix and `x_i` its `i`-th column. Non-negativity turns the
L1 term into a plain linear penalty, so this is a non-negative elastic net with an exact
coordinate-wise minimiser:

```text
w_k <- max(0, (P_ik - sum_{j != k} w_j P_jk - lambd) / (P_kk + beta))
```

with `P = X.T @ X`. The implementation builds `P` once by sparse accumulation in a
fixed order, keeps the residuals `r_k = P_ik - sum_j w_j P_kj` incrementally, restricts
each item to candidates with `P_ik >= lambd` (provably exact for non-negative data), and
alternates active-set passes with full verification passes. Column `i` of the result is
solved by one thread from start to finish, which is why the answer does not depend on
how many threads there are.

The full derivation — including why the candidate restriction is exact, the KKT bound
the stopping rule guarantees, and where this differs from the original paper — is in
[`docs/algorithm.md`](https://github.com/rmnigm/fastslim/blob/main/docs/algorithm.md).

## Development

```bash
uv sync                        # dev group; builds the extension
uv run pytest -q -m "not slow" # 180 fast Python tests, a few seconds
cargo test                     # Rust unit tests
```

See [`CONTRIBUTING.md`](https://github.com/rmnigm/fastslim/blob/main/CONTRIBUTING.md)
for the repo layout, linting, benchmarks and release steps, and
[`CHANGELOG.md`](https://github.com/rmnigm/fastslim/blob/main/CHANGELOG.md) for what
changed between releases.

## License

Apache-2.0. See
[`LICENSE`](https://github.com/rmnigm/fastslim/blob/main/LICENSE).
