# Contributing to fastslim

`fastslim` is a Rust solver behind a thin Python API, with the correctness argument
written down in [`docs/algorithm.md`](docs/algorithm.md) and the public surface in
[`docs/api.md`](docs/api.md). Changes that alter the numbers the solver returns should
say why in those terms.

## Setup

You need a [Rust toolchain](https://rustup.rs) and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/rmnigm/fastslim
cd fastslim
uv sync              # dev group: pytest, ruff, pre-commit, hypothesis
uv run pre-commit install
```

`uv sync` builds the extension module through maturin and installs the package in
editable mode, so `uv run python -c "import fastslim"` works immediately.

Add `--group bench` when you want the benchmarks or the MovieLens tests:

```bash
uv sync --group bench   # implicit, h5py, rich, tqdm on top of the dev group
```

## Repo layout

| Path | What lives there |
| --- | --- |
| `src/gram.rs` | Builds the Gram matrix `P = X.T @ X` by sparse accumulation. Deterministic by construction; see §8 of the algorithm doc |
| `src/solver.rs` | Per-item non-negative coordinate descent: candidate selection, residual maintenance, active set, `solve_slim_csr`. Pure Rust, no pyo3 |
| `src/lib.rs` | The pyo3 bindings. Validates and converts, then calls `solve_slim_csr`; no numerics |
| `src/testing.rs` | Test-only helpers (`csr_from_dense`, `random_binary`) |
| `src/gram/tests.rs`, `src/solver/tests.rs` | The Rust unit tests for the module next to them |
| `python/fastslim/api.py` | `fit`, `predict`, `recommend` and the single call site of the extension (`solve`) |
| `python/fastslim/validation.py` | Input and hyperparameter checks shared by the functional API and the estimator |
| `python/fastslim/estimator.py` | The `SLIM` class and `NotFittedError` |
| `python/fastslim/metrics.py` | `precision_at_k`, `recall_at_k`, `ndcg_at_k` |
| `python/fastslim/native.pyi` | Stubs for the compiled module — keep in sync with `src/lib.rs` |
| `tests/` | The Python suite, plus `conftest.py` with fixtures and a KKT checker |
| `docs/algorithm.md`, `docs/api.md` | The correctness argument and the public API reference |
| `benchmarks/benchmark.py` | The benchmark CLI |

Keep the numerics in `gram.rs` and `solver.rs` and the conversions in `lib.rs`. On the
Python side, everything that talks to the extension goes through `api.solve`, so a
change to the binding's return tuple has one place to land.

## Tests

```bash
uv run pytest -q -m "not slow"   # 210 tests, a few seconds
uv run pytest -q                 # adds the MovieLens integration tests
cargo test                       # 20 Rust unit tests
```

The `slow` marker covers the MovieLens tests, which need the `bench` group (for
`implicit`) and download the dataset to `~/implicit_data` on first run. They are skipped
rather than failed when `implicit` is missing.

`cargo test` builds a test binary that links against libpython, so the
`extension-module` pyo3 feature stays off by default. If the build script cannot find an
interpreter with a shared library, point it at the project venv:

```bash
PYO3_PYTHON=$(pwd)/.venv/bin/python cargo test
```

Correctness tests assert against something independent of the Rust code — the KKT
conditions, the naive dense solver in `tests/conftest.py`, or arithmetic worked out by
hand — so that a self-consistent regression still gets caught. New solver behaviour
should come with a test in that style, not one that records whatever the code currently
prints.

## Lint and format

```bash
uv run ruff check python/ benchmarks/ tests/
uv run ruff format --check python/ benchmarks/ tests/
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

`ruff` is pinned in the `dev` dependency group and the pre-commit hook `rev` is kept at
the same version, so local runs, pre-commit and CI never disagree. `pre-commit install`
wires all of the above (ruff with `--fix`, ruff-format, `cargo fmt`, `cargo clippy`) into
`git commit`; `uv run pre-commit run --all-files` runs them over the whole tree.

## Rebuilding after Rust changes

`uv run pytest` does **not** rebuild the extension. After editing anything under `src/`:

```bash
uv sync --reinstall-package fastslim
```

Python-only edits need nothing — the package is installed editable.

## Benchmarks

```bash
uv sync --group bench
uv run python benchmarks/benchmark.py --help
uv run python benchmarks/benchmark.py --dataset synthetic          # no download
uv run python benchmarks/benchmark.py --dataset 1m --baseline als --markdown
```

`--dataset` picks `100k`, `1m` or `synthetic`; `--baseline als` adds an `implicit`
ALS run for comparison; `--markdown` writes a GitHub table to stdout with all progress
output on stderr, which is how the README tables are produced. Solver knobs
(`--lambd`, `--beta`, `--max-iter`, `--tol`, `--threads`), the metric cut-off (`--k`)
and the split seed (`--seed`) are all exposed.

The peak-RSS column is a process-wide high-water mark measured across sequential fits,
not an isolated per-model number — a model fitted after a hungrier one can report 0.
Treat it as a floor, and do not quote it as a per-model figure.

When you update a benchmark number in the README, quote the command that produced it and
the machine it ran on.

## Releasing

1. Bump `version` in `Cargo.toml`. That is the only place it lives; `pyproject.toml`
   declares it `dynamic` and maturin reads it from the crate.
2. Move the `Unreleased` entries in `CHANGELOG.md` into a dated section for the new
   version and add a fresh `Unreleased` header.
3. Merge to `main` and confirm CI is green.
4. Tag and push the tag:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

The `Publish` workflow triggers on `v*` tags. It builds `abi3` wheels for Linux x86_64
and aarch64 (manylinux), macOS universal2 and Windows x64 plus an sdist, smoke-tests
each wheel in a clean venv, and uploads everything to PyPI through trusted publishing —
no API token in the repo. It can also be started manually with `workflow_dispatch`.

CI (`.github/workflows/ci.yml`) runs on every push and pull request: Rust lint, Rust
tests, Python lint, the fast Python suite on Linux/macOS/Windows against Python 3.10 and
3.14, and the slow integration job. Wheels are only built on tags, so a red publish run
after a green CI run usually means a packaging problem rather than a code one.
