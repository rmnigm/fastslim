# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-06

A near-total rewrite of the solver and the Python surface. The solver now returns the
actual optimum of the objective it documents, does so deterministically, and does it
roughly an order of magnitude faster; the package around it grew a prediction and
evaluation API, real input validation, and published wheels.

### Breaking

- **Results differ from 0.1.x at identical hyperparameters.** The old solver stopped at
  a non-optimal point: it dropped active-set coordinates permanently and skipped whole
  items on an invalid test. `fit` now solves the problem to KKT optimality, so expect a
  different — and denser — `W`, and better ranking metrics. Re-tune `lambd` rather than
  assuming an old value still gives the same sparsity.
- **New defaults: `max_iter=1000`, `tol=1e-4`** (0.1.x: `max_iter=100`, `tol=1e-6`).
  Popular items with near-identical interaction columns are ill-conditioned, and the old
  budget left 2,503 of the 3,953 MovieLens 1M items truncated. With the new defaults
  every ML-1M item converges in about 12 s on a laptop; an item that still runs out of
  budget is reported through `ConvergenceWarning` instead of silently returned.
- **The returned container mirrors the input.** A SciPy sparse *array* (`csr_array`,
  `coo_array`, ...) now yields a `csr_array`; sparse matrices, dense arrays and
  array-likes yield a `csr_matrix`. 0.1.x always returned a `csr_matrix`.
- **Inputs are validated and bad ones raise.** A non-2-D matrix, a negative or
  non-finite value, or an out-of-range hyperparameter raises `ValueError`; a
  hyperparameter of the wrong type (`max_iter=1.5`, `max_iter=True`) raises `TypeError`.
  0.1.x accepted these and returned nonsense or aborted.
- **The private extension API changed.** `fastslim._slim_rs.solve_slim` now takes the
  CSR arrays as keyword arguments and returns `(indptr, indices, data, n_passes,
  converged)`: `W` in CSC layout plus per-item pass counts and convergence flags. It is
  private; call `fastslim.fit`.
- **Tests are no longer shipped in the wheel.** The distribution contains
  `python/fastslim/` only; the suite lives in `tests/` at the repo root.
- The package is Apache-2.0 throughout. 0.1.x metadata disagreed with itself about the
  license (see Fixed).

### Added

- `fastslim.predict` (dense scores for one user or a batch, optionally masking seen
  items) and `fastslim.recommend` (top-`k` item ids, ranked best first, with
  deterministic tie-breaking and a `batch_size` that genuinely bounds peak memory).
- `fastslim.SLIM`, a scikit-learn-flavoured estimator (`fit`, `predict`, `recommend`,
  `get_params`, `set_params`, `weights_`, `n_items_`, `n_passes_`, `converged_`,
  `__repr__`) that does not depend on scikit-learn, plus `fastslim.NotFittedError`.
- `fastslim.metrics` with `precision_at_k`, `recall_at_k` and `ndcg_at_k`, accepting
  either a score array from `predict` or a ranked-id array from `recommend`, and
  skipping users with no held-out items instead of scoring them zero.
- `fastslim.ConvergenceWarning`, emitted by `fit` and `SLIM.fit` when some items exhaust
  `max_iter` before reaching `tol`, and the per-item `n_passes_` / `converged_`
  diagnostics on the estimator.
- A `tol` parameter on `fit`: the stopping tolerance was hard-coded in 0.1.x.
- Input flexibility: any SciPy sparse format, sparse arrays, dense `ndarray`s and nested
  sequences are accepted, `int32` and `int64` index arrays are consumed as-is, and the
  caller's matrix is never mutated.
- Type stubs (`_slim_rs.pyi`) and a `py.typed` marker, so `fit`'s signature and `W`'s
  layout are visible to type checkers.
- `fastslim.__version__`, single-sourced from `Cargo.toml` through maturin.
- A test suite: 183 Python tests (property-based tests included, plus a MovieLens
  integration test behind `-m slow`) and 16 Rust unit tests, checking the solver against
  the KKT conditions and an independent dense reference solver rather than against
  itself.
- `benchmarks/benchmark.py`, a CLI covering MovieLens 100k/1M and a synthetic matrix,
  with an optional `implicit` ALS baseline and markdown output.
- Documentation: a rewritten `README.md`, `docs/algorithm.md` with the full derivation,
  this changelog and `CONTRIBUTING.md`.
- CI on Linux, macOS and Windows across Python 3.10 and 3.14, with `cargo fmt`,
  `cargo clippy -D warnings`, `cargo test` and `ruff` gates and a pre-commit config; and
  a publish workflow that builds wheels and an sdist on `v*` tags and uploads them to
  PyPI via trusted publishing.

### Fixed

- **Non-determinism.** The Gram matrix was accumulated through a hash map of item pairs,
  so floating-point summation order — and therefore the last bits of every weight —
  depended on thread count and scheduling. It is now built by a fixed-order CSC x CSR
  sparse accumulation with no hash maps, and each item is solved start to finish by one
  thread. `fit` is bit-identical across runs and across any `n_threads`.
- **Permanently dropped active-set coordinates.** A weight that reached zero was removed
  from the active set for good and could never re-enter, so the solver converged to a
  point that was not the optimum. Full verification passes now alternate with active-set
  passes, and a fit only reports convergence after a full pass over every candidate.
- **The invalid `P_ii < lambd^2` item skip.** 0.1.x zeroed an entire item when its Gram
  diagonal fell below `lambd^2`. The test does not follow from the objective and
  discarded items with a genuinely nonzero solution. Removed; the remaining
  `P_ik >= lambd` candidate filter is proved exact for non-negative data in
  [`docs/algorithm.md`](https://github.com/rmnigm/fastslim/blob/main/docs/algorithm.md).
- **Nonzero diagonal on duplicate CSR entries.** The Gram diagonal was computed
  separately from the off-diagonal entries, so a matrix with repeated `(row, column)`
  entries produced a `P_kk` inconsistent with the rest of row `k` and left nonzero
  values on `W`'s diagonal. Both now come out of the same accumulator, and duplicates
  are summed the way SciPy sums them.
- **License metadata mismatch.** The project advertised MIT in some places and
  Apache-2.0 in others. It is Apache-2.0, now stated consistently in `pyproject.toml`,
  the classifier list and `LICENSE`.
- **Missing wheels.** No wheels were published for 0.1.x, so every install had to build
  the crate from source. 0.2.0 ships `abi3` wheels for Linux x86_64 and aarch64
  (manylinux), macOS universal2 and Windows x64, plus an sdist.
- The `README` documented the L2 term as `beta * sum(w^2)`; the code always used the
  SLIM paper's `(beta / 2) * sum(w^2)`, and the documentation now matches.

### Performance

Measured with 12 threads on Apple Silicon, median of 3 runs:

| Workload | 0.1.x | 0.2.0 |
| --- | --- | --- |
| Synthetic 20,000 x 3,000, 1% density, `max_iter=20` | 65.3 s, 1455 MiB peak RSS | 7.0 s, 757 MiB |
| MovieLens 1M (6,041 x 3,953, 1.0M nnz), `max_iter=15` | 25.6 s, 3902 MiB peak RSS | 0.85 s, 400 MiB |

The same synthetic fit takes 51.6 s on a single thread, so most of the speedup is
algorithmic rather than parallelism:

- The Gram matrix is built by sparse accumulation over a CSC view of `X` instead of a
  hash map keyed by item pairs, which also removes the hash map's allocation churn.
- Coordinate descent maintains residuals incrementally and updates them only when a
  weight actually moves, instead of recomputing partial correlations per visit.
- Candidate membership is a dense item-indexed lookup reused across items, not a search.
- The release profile enables fat LTO and a single codegen unit.

## [0.1.3] - [0.1.1]

Initial releases: a single `fastslim.fit` function and no published wheels, so installs
built the crate from source. Superseded by 0.2.0, whose solver returns different (and
correct) weights.

[Unreleased]: https://github.com/rmnigm/fastslim/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/rmnigm/fastslim/releases/tag/v0.2.0
