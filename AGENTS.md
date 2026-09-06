# AGENTS.md

Operating notes for coding agents working on `fastslim`. Human-facing documentation is
in `README.md`, `docs/`, and `CONTRIBUTING.md`; this file holds what an agent needs to
change the code safely.

## What the project is

A Rust implementation of SLIM (sparse linear item-item recommendation) exposed to
Python through pyo3 and maturin. One compiled module, `fastslim.native`, one solver
entry point, and a thin Python layer for validation, prediction, an estimator and
metrics. Published on PyPI as `fastslim`; the version is read from `Cargo.toml`.

## Layout

| Path | Role |
| --- | --- |
| `Cargo.toml` | Crate `slim`, lib name `native`, pyo3 `abi3-py310`, `extension-module` as an opt-in feature (enabled by maturin, off for `cargo test`) |
| `src/lib.rs` | pyo3 bindings and argument validation only |
| `src/gram.rs` | Gram matrix `P = XᵀX` by per-item CSC×CSR accumulation, fixed summation order |
| `src/solver.rs` | Per-item non-negative coordinate descent, active set, convergence check, CSC assembly |
| `src/gram/tests.rs`, `src/solver/tests.rs`, `src/testing.rs` | Rust unit tests and shared test helpers (in-crate because the crate is a `cdylib`) |
| `python/fastslim/__init__.py` | Public exports via `__all__` and `__version__` |
| `python/fastslim/api.py` | `fit`, `predict`, `recommend`; `solve` is the single call into `native`; `fit_with_diagnostics` is shared with the estimator |
| `python/fastslim/validation.py` | Input coercion to canonical CSR float64 and parameter checks; never mutates the caller's matrix |
| `python/fastslim/estimator.py` | `SLIM` and `NotFittedError` |
| `python/fastslim/metrics.py` | `precision_at_k`, `recall_at_k`, `ndcg_at_k` |
| `python/fastslim/native.pyi`, `py.typed` | Type stub for the extension; both ship in the wheel |
| `tests/` | pytest suite; `conftest.py` holds fixtures and the KKT checker |
| `benchmarks/benchmark.py` | CLI benchmark against implicit ALS, markdown output |
| `docs/api.md`, `docs/algorithm.md` | Parameter reference; derivation and guarantees |
| `.github/workflows/ci.yml` | Lint, Rust tests, Python 3.10/3.14 on Linux, macOS, Windows, slow integration job |
| `.github/workflows/publish.yml` | Wheels and sdist on `v*` tags, PyPI trusted publishing |

## Commands

```bash
uv sync                                   # dev group (pytest, ruff, hypothesis, pre-commit); builds the extension
uv sync --group bench                     # adds implicit, h5py, rich, tqdm for the slow test and the benchmark
uv sync --reinstall-package fastslim      # REQUIRED after any change under src/, Cargo.toml or [tool.maturin]
uv run pytest -q -m "not slow"            # fast suite, well under a second
uv run pytest -q                          # includes the MovieLens 100k test (downloads to ~/implicit_data once)
uv run ruff check python/ tests/ benchmarks/ && uv run ruff format --check python/ tests/ benchmarks/
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test
uv run python benchmarks/benchmark.py --dataset 1m --baseline als --markdown
uv build && unzip -l dist/*.whl           # wheel must contain native.pyi and py.typed and no tests
```

If `cargo test` fails to link, point pyo3 at an interpreter with a shared library:
`PYO3_PYTHON=$(pwd)/.venv/bin/python cargo test`. After renaming the extension, delete
any stale `python/fastslim/*.so` by hand; uv does not remove it.

README code blocks are part of the test surface: extract every ```python fence, run
them in order in one namespace, and they must succeed.

## Contracts

These are relied on by tests, docs and users. Do not change them without updating all
three.

- Objective per item `i`: `0.5·‖x_i − Xw‖² + λ·Σw + (β/2)·Σw²`, `w ≥ 0`, `w_i = 0`.
  The L2 term is `β/2`, matching the SLIM paper.
- `native.solve_slim(data, indices, indptr, n_rows, n_cols, lambd, beta, max_iter, tol,
  n_threads)` returns `(indptr, indices, data, n_passes, converged)`. The first three
  arrays are `W` in CSC layout: segment `i` holds the neighbours of target item `i`,
  sorted ascending, weights strictly positive, so `W[k, i] = w_ik` and
  `scores = X @ W`. `n_passes` is `int64` and `converged` is `bool`, one entry per item.
- Determinism: output is byte-identical for any `n_threads` and across runs. Keep it
  that way: no parallel floating-point reductions, per-item work only, Gram rows
  accumulated in a fixed order. `tests/test_determinism.py` enforces it.
- Convergence guarantee: an item flagged `converged` satisfies the KKT conditions to
  within `(P_kk + β)·tol` per coordinate. The final check evaluates unclipped would-be
  steps at the final residuals; do not replace it with the clipped step.
- The candidate filter `P_ik ≥ λ` is exact only for non-negative data, which is why
  validation rejects negative or non-finite values on both the Python and Rust side.
- `max_iter` counts active-set and full passes together; `tol = 0` means exactly
  `max_iter` passes. Items that hit the budget are returned as they stand and flagged.
- Defaults `lambd=0.5, beta=0.5, max_iter=1000, tol=1e-4` live in four places that must
  agree: `api.fit`, `estimator.DEFAULTS` and the `SLIM` signature, the pyo3 signature
  in `src/lib.rs`, and `SlimParams::default()` in `src/solver.rs`.
- Errors: `ValueError` for bad values or structure, `TypeError` for wrong types,
  `ConvergenceWarning` (a `UserWarning`) when items exhaust `max_iter`. The pytest
  config turns that warning into an error; a test that expects truncation must say so
  with `pytest.warns` or a `filterwarnings` mark.
- Return container mirrors the input: SciPy sparse array in, `csr_array` out;
  anything else, `csr_matrix`.

## Style rules

1. Imports at module top level only. Optional dependencies use `try/except ImportError`
   at the top and fail at the point of use with a clear message.
2. No module docstrings and no multi-line comment blocks. One docstring of one or two
   sentences per public class and function. Explanations go to `docs/`.
3. Docs describe behaviour only: no change history, no design rationale. History
   belongs in `CHANGELOG.md`, as user-facing entries.
4. No leading and no trailing underscores in file, class, function or attribute names.
   Internal underscores (snake_case) are fine. Language-required dunders are exempt.
   The public surface is defined by `__all__`, not by underscores.
5. Rust unit tests live in `src/<module>/tests.rs`, never inline in the source file.
6. Python tests are parametrized and compact: `pytest.mark.parametrize` with readable
   ids, shared setup in `conftest.py`, one behaviour per test.
7. ruff is configured in `pyproject.toml` with `E F I UP B W D N PT`, numpy docstring
   convention, and `D100 D103 D104 D105 D107` ignored. Keep it green.

## Testing philosophy

Correctness evidence comes from three sources, and only these: checks against the
specification (`kkt_violation` in `tests/conftest.py` evaluates the optimality
conditions of the objective directly from `X` and the returned `W`), hand-computed
fixed values, and end-to-end ranking metrics on MovieLens 100k with a stored baseline.
Do not add a second implementation of the algorithm as an oracle; implementation
versus implementation proves nothing. Property tests use hypothesis with small
matrices and generous `max_iter`.

## Git and review conventions

- `main` is protected by the `protect-main` ruleset: pull request required, all ten CI
  checks required with the branch up to date, review threads resolved, no force push,
  no deletion, no bypass.
- Work on a feature branch; commit per logical step; commit messages are a subject line
  plus at most three short sentences.
- PR descriptions are functional: what the change introduces, then a short testing
  section.
- CI runs on pushes to `main` and on pull requests. Workflow runs on pull requests from
  outside contributors wait for maintainer approval.
- Agent worktrees are created from `main`; reset onto the feature branch before doing
  anything (`git reset --hard <branch>` while the agent branch has no commits).

## Release

Bump `version` in `Cargo.toml`, add the user-facing entry to `CHANGELOG.md`, merge to
`main`, tag `vX.Y.Z` and push the tag. `publish.yml` builds wheels on native runners
(Linux x86_64 and aarch64, macOS universal2, Windows x64) plus an sdist, smoke-tests
each wheel in a fresh venv, and publishes through the PyPI trusted publisher configured
for `rmnigm/fastslim` with workflow `publish.yml`.

## Reference numbers

Useful for spotting regressions without a full benchmark (12-thread Apple Silicon).

| Measurement | Value |
| --- | --- |
| ML-1M fit, `lambd=beta=0.5`, `max_iter=15` | about 0.85 s, 400 MB peak |
| ML-1M fit with defaults, all items converged | about 12 s |
| ML-100k benchmark defaults | precision@10 0.3391, recall@10 0.2236, ndcg@10 0.4069 |
| Seeded 2000×300 binary matrix, `lambd=beta=1` | 21,821 nonzeros, identical for any thread count |
| Fast pytest suite | 200 tests, under one second |
