# fastslim

[![PyPI](https://img.shields.io/pypi/v/fastslim.svg)](https://pypi.org/project/fastslim/)
[![CI](https://img.shields.io/github/actions/workflow/status/rmnigm/fastslim/ci.yml?branch=main&label=CI)](https://github.com/rmnigm/fastslim/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/fastslim.svg)](https://pypi.org/project/fastslim/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/rmnigm/fastslim/blob/main/LICENSE)

SLIM ([Sparse Linear Methods](https://ieeexplore.ieee.org/document/6137254), Ning &
Karypis 2011) learns a sparse, non-negative item-item weight matrix `W` from a
user-item matrix `X` and scores users with one sparse product, `X @ W`. Every
recommendation is explainable: "because you interacted with these items".

`fastslim` is a Rust implementation behind a small Python API. The solver is exact
(the returned weights satisfy the optimality conditions of the objective below),
deterministic (bit-identical output for any thread count) and fast: MovieLens 1M fits
in about a second.

## Install

```bash
pip install fastslim
```

Wheels for Linux (x86_64, aarch64), macOS and Windows, Python 3.10 and newer. Runtime
dependencies are NumPy and SciPy.

## Usage

```python
import numpy as np
from scipy import sparse

import fastslim

rng = np.random.default_rng(0)
X = sparse.csr_matrix((rng.random((2000, 300)) < 0.03).astype(np.float64))  # users x items

W = fastslim.fit(X, lambd=2.0, beta=2.0)      # items x items, sparse, zero diagonal
top = fastslim.recommend(W, X[:5], k=10)       # top-10 unseen items for five users
scores = fastslim.predict(W, X[0])             # raw scores for one user
```

The same solver as an estimator:

```python
from fastslim import SLIM

model = SLIM(lambd=2.0, beta=2.0).fit(X)
model.recommend(X[0], k=5)
model.weights, model.n_passes, model.converged  # fitted state
```

`fit` takes any SciPy sparse matrix or array, or a dense array; values must be finite
and non-negative. `fastslim.metrics` provides `precision_at_k`, `recall_at_k` and
`ndcg_at_k` against a held-out sparse matrix. The full reference is in
[`docs/api.md`](https://github.com/rmnigm/fastslim/blob/main/docs/api.md).

## Hyperparameters

- **`lambd`**, the L1 penalty, is compared against co-occurrence counts: item `k` can
  enter item `i`'s model only if `(X.T @ X)[i, k] >= lambd`. On MovieLens-sized data
  useful values are single digits. Raise it for a sparser `W`.
- **`beta`**, the L2 penalty, shrinks weights and conditions the problem; any
  `beta > 0` makes the solution unique.
- **`max_iter`** and **`tol`** stop each item once a full pass moves no weight by `tol`.
  The defaults (`1000`, `1e-4`) converge all of MovieLens 1M. Items that run out of
  budget raise `fastslim.ConvergenceWarning`; `SLIM.converged` tells which ones.
- **`n_threads`** changes speed only. The result does not depend on it.

## Performance

MovieLens, 80/20 split, `k=10`, against implicit's ALS with 64 factors. SLIM uses
`lambd=2`, `beta=2`, `max_iter=50`.

| Dataset | Model | Fit time | precision@10 | recall@10 | ndcg@10 |
| --- | --- | --- | --- | --- | --- |
| ML-100k | fastslim | 0.17 s | **0.3391** | **0.2236** | **0.4069** |
| ML-100k | implicit ALS | 0.58 s | 0.2878 | 0.2017 | 0.3433 |
| ML-1M | fastslim | 2.55 s | **0.3620** | **0.1652** | **0.4100** |
| ML-1M | implicit ALS | 4.45 s | 0.3465 | 0.1621 | 0.3915 |

Reproduce with `uv sync --group bench` and
`uv run python benchmarks/benchmark.py --dataset 1m --baseline als --markdown`.

## How it works

For every item `i` independently, the solver minimises

```text
0.5 * ||x_i - X w||^2 + lambd * sum_k w_k + (beta / 2) * sum_k w_k^2
subject to  w >= 0,  w_i = 0
```

by non-negative coordinate descent on the Gram matrix `P = X.T @ X`: only neighbours
with `P_ik >= lambd` can be non-zero, residuals are updated incrementally, and an active
set is iterated until a full verification pass confirms optimality. `P` is accumulated
in a fixed order and each item is solved by a single thread, which is what makes the
output deterministic. The derivation is in
[`docs/algorithm.md`](https://github.com/rmnigm/fastslim/blob/main/docs/algorithm.md).

## Development

```bash
uv sync                          # builds the extension, installs dev tools
uv run pytest -q -m "not slow"   # Python tests
cargo test                       # Rust tests
```

[`CONTRIBUTING.md`](https://github.com/rmnigm/fastslim/blob/main/CONTRIBUTING.md) covers
layout, linting, benchmarks and releases;
[`CHANGELOG.md`](https://github.com/rmnigm/fastslim/blob/main/CHANGELOG.md) lists what
changed between versions.

## License

Apache-2.0. See [`LICENSE`](https://github.com/rmnigm/fastslim/blob/main/LICENSE).
