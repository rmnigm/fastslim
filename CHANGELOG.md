# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-06

### Added

- `fastslim.predict` and `fastslim.recommend`, with `exclude_seen` and a `batch_size`
  that bounds peak memory.
- `fastslim.SLIM`, a scikit-learn-flavoured estimator, and `fastslim.NotFittedError`.
- `fastslim.metrics` with `precision_at_k`, `recall_at_k` and `ndcg_at_k`.
- `fastslim.ConvergenceWarning`, plus the per-item `n_passes` and `converged`
  diagnostics on the estimator.
- A `tol` parameter on `fit`; the stopping tolerance was hard-coded in 0.1.x.
- Type stubs and a `py.typed` marker; `fastslim.__version__`.

### Changed

- **Results differ from 0.1.x at identical hyperparameters.** `fit` now solves to KKT
  optimality, so expect a different and denser `W`, and better ranking metrics. Re-tune
  `lambd`.
- **`fit` is deterministic**: bit-identical output across runs and across any
  `n_threads`.
- **New defaults: `max_iter=1000`, `tol=1e-4`** (0.1.x: `max_iter=100`, `tol=1e-6`).
- **The returned container mirrors the input**: a SciPy sparse *array* now yields a
  `csr_array`; everything else yields a `csr_matrix`.
- **Bad input raises.** A non-2-D matrix, a negative or non-finite value or an
  out-of-range hyperparameter raises `ValueError`; a wrong-typed hyperparameter raises
  `TypeError`. Any SciPy sparse format, sparse arrays, dense arrays and nested sequences
  are accepted, and the caller's matrix is never mutated.
- The package is Apache-2.0 throughout; 0.1.x metadata disagreed with itself.
- Tests are no longer shipped in the wheel.

### Performance

12 threads on Apple Silicon, median of 3 runs.

| Workload | Fit time 0.1.x | Fit time 0.2.0 | Peak RSS 0.1.x | Peak RSS 0.2.0 |
| --- | --- | --- | --- | --- |
| Synthetic 20,000 x 3,000, 1% density, `max_iter=20` | 65.3 s | 7.0 s | 1455 MiB | 757 MiB |
| MovieLens 1M (6,041 x 3,953, 1.0M nnz), `max_iter=15` | 25.6 s | 0.85 s | 3902 MiB | 400 MiB |

### Packaging

- `abi3` wheels for Linux x86_64 and aarch64 (manylinux), macOS universal2 and Windows
  x64, plus an sdist. 0.1.x published no wheels.

## [0.1.3] - [0.1.1]

Initial releases: a single `fastslim.fit` function and no published wheels.

[Unreleased]: https://github.com/rmnigm/fastslim/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/rmnigm/fastslim/releases/tag/v0.2.0
