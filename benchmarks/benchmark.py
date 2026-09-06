"""Benchmark fastslim's SLIM solver, optionally against implicit's ALS.

Examples
--------
    python benchmarks/benchmark.py --dataset 100k
    python benchmarks/benchmark.py --dataset 1m --baseline als --markdown
    python benchmarks/benchmark.py --dataset synthetic --baseline none

The MovieLens variants need the ``bench`` dependency group (``uv sync --group
bench``) and download to ``~/implicit_data`` on first use.  ``synthetic`` needs
nothing beyond fastslim itself, so the script is always runnable.

Memory is reported as the growth in the process's peak resident set size across
a model's fit.  Because ``ru_maxrss`` is a high-water mark that never falls, a
model fitted after a hungrier one can show 0 -- the number is a floor on what
that fit cost, not an isolated measurement.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import fastslim
import numpy as np
from fastslim import metrics
from scipy import sparse

try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
except ImportError:  # pragma: no cover - rich is optional
    Console = Progress = SpinnerColumn = TextColumn = Table = None

try:
    from implicit.als import AlternatingLeastSquares
    from implicit.datasets.movielens import get_movielens
    from implicit.evaluation import train_test_split
except ImportError:  # pragma: no cover - the bench group is optional
    AlternatingLeastSquares = get_movielens = train_test_split = None

# ru_maxrss is bytes on macOS and kibibytes on Linux.
_RSS_SCALE = 1.0 if sys.platform == "darwin" else 1024.0

SCORING_BATCH = 1000


def peak_rss_mb() -> float:
    """Peak resident set size of this process so far, in MiB (0 if unknown)."""
    if resource is None:
        return 0.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_SCALE / 1024**2


@dataclass
class Result:
    """One model's timing, memory and accuracy on one dataset."""

    name: str
    fit_seconds: float
    rss_growth_mb: float
    model_size: str
    scores: dict[str, float] = field(default_factory=dict)
    extra: dict[str, str] = field(default_factory=dict)


class Reporter:
    """Progress and logging, degrading to plain prints when rich is missing.

    Everything it writes goes to stderr, so ``--markdown`` can pipe a clean
    table out of stdout.
    """

    def __init__(self, use_rich: bool = True) -> None:
        self.console = (
            Console(stderr=True) if (use_rich and Console is not None) else None
        )

    def log(self, message: str) -> None:
        if self.console is not None:
            self.console.print(message)
        else:
            print(message, file=sys.stderr)

    @contextmanager
    def step(self, description: str) -> Iterator[None]:
        if self.console is None:
            print(f"{description}...", file=sys.stderr, flush=True)
            yield
            return
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            progress.add_task(description, total=None)
            yield


@contextmanager
def timed() -> Iterator[dict[str, float]]:
    """Wall-clock seconds and peak-RSS growth for the enclosed block."""
    result: dict[str, float] = {}
    rss_before = peak_rss_mb()
    start = time.perf_counter()
    yield result
    result["seconds"] = time.perf_counter() - start
    result["rss_growth_mb"] = max(peak_rss_mb() - rss_before, 0.0)


def load_dataset(name: str, seed: int) -> sparse.csr_matrix:
    """Return a binary ``(users x items)`` CSR matrix."""
    if name == "synthetic":
        rng = np.random.default_rng(seed)
        n_users, n_items, per_user = 20_000, 2_000, 20
        rows = np.repeat(np.arange(n_users), per_user)
        cols = rng.integers(0, n_items, size=rows.size)
        matrix = sparse.csr_matrix(
            (np.ones(rows.size), (rows, cols)), shape=(n_users, n_items)
        )
        matrix.data[:] = 1.0
        return matrix

    _, ratings = get_movielens(name)
    # get_movielens returns (items x users); implicit feedback means "rated at
    # all", so binarise rather than keeping the star ratings.
    return (ratings.T.tocsr() > 0).astype(np.float64)


def split(
    matrix: sparse.csr_matrix, seed: int
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """80/20 split, using implicit's splitter when it is available."""
    if train_test_split is None:
        rng = np.random.default_rng(seed)
        coo = matrix.tocoo()
        keep = rng.random(coo.nnz) < 0.8
        parts = [
            sparse.csr_matrix(
                (coo.data[mask], (coo.row[mask], coo.col[mask])), shape=matrix.shape
            )
            for mask in (keep, ~keep)
        ]
        return parts[0], parts[1]

    train, test = train_test_split(matrix, train_percentage=0.8, random_state=seed)
    return train.tocsr(), test.tocsr()


def top_k_batched(
    scorer: Callable[[int, int], np.ndarray], n_users: int, k: int
) -> np.ndarray:
    """Rank users in batches, keeping only ``k`` items each.

    ``scorer`` maps a user range to a dense score block.  Used for models that
    have no item-item matrix to hand to :func:`fastslim.recommend`.
    """
    top = np.empty((n_users, k), dtype=np.int64)
    for start in range(0, n_users, SCORING_BATCH):
        stop = min(start + SCORING_BATCH, n_users)
        block = scorer(start, stop)
        n_items = block.shape[1]
        candidates = np.argpartition(block, n_items - k, axis=1)[:, n_items - k :]
        candidate_scores = np.take_along_axis(block, candidates, axis=1)
        order = np.lexsort((candidates, -candidate_scores), axis=-1)
        top[start:stop] = np.take_along_axis(candidates, order, axis=1)
    return top


def score_all(top: np.ndarray, test: sparse.csr_matrix, k: int) -> dict[str, float]:
    return {
        f"precision@{k}": metrics.precision_at_k(top, test, k=k),
        f"recall@{k}": metrics.recall_at_k(top, test, k=k),
        f"ndcg@{k}": metrics.ndcg_at_k(top, test, k=k),
    }


def run_slim(
    train: sparse.csr_matrix,
    test: sparse.csr_matrix,
    args: argparse.Namespace,
    reporter: Reporter,
) -> Result:
    model = fastslim.SLIM(
        lambd=args.lambd,
        beta=args.beta,
        max_iter=args.max_iter,
        tol=args.tol,
        n_threads=args.threads,
    )
    with reporter.step("Fitting SLIM"), timed() as timing, warnings.catch_warnings():
        # Truncation is reported as a table row instead of a warning.
        warnings.simplefilter("ignore", fastslim.ConvergenceWarning)
        model.fit(train)
    weights = model.weights_

    with reporter.step("Scoring SLIM"):
        top = fastslim.recommend(weights, train, k=args.k, batch_size=SCORING_BATCH)

    n_unconverged = int(np.count_nonzero(~model.converged_))
    return Result(
        name="fastslim SLIM",
        fit_seconds=timing["seconds"],
        rss_growth_mb=timing["rss_growth_mb"],
        model_size=f"{weights.nnz:,} nonzeros",
        scores=score_all(top, test, args.k),
        extra={
            "Items hitting max_iter": f"{n_unconverged:,} / {model.n_items_:,}",
            "Median passes per item": f"{np.median(model.n_passes_):.0f}",
        },
    )


def run_als(
    train: sparse.csr_matrix,
    test: sparse.csr_matrix,
    args: argparse.Namespace,
    reporter: Reporter,
) -> Result:
    model = AlternatingLeastSquares(
        factors=args.factors,
        regularization=0.01,
        iterations=args.max_iter,
        use_gpu=False,
        random_state=args.seed,
    )
    with reporter.step("Fitting ALS"), timed() as timing:
        model.fit(train, show_progress=False)

    user_factors = np.asarray(model.user_factors)
    item_factors = np.asarray(model.item_factors)

    def scorer(start: int, stop: int) -> np.ndarray:
        block = (user_factors[start:stop] @ item_factors.T).astype(np.float64)
        # Mask the training history, exactly as fastslim.predict does.
        lengths = np.diff(train.indptr[start : stop + 1])
        rows = np.repeat(np.arange(stop - start), lengths)
        seen = train.indices[train.indptr[start] : train.indptr[stop]]
        block[rows, seen] = -np.inf
        return block

    with reporter.step("Scoring ALS"):
        top = top_k_batched(scorer, train.shape[0], args.k)

    return Result(
        name="implicit ALS",
        fit_seconds=timing["seconds"],
        rss_growth_mb=timing["rss_growth_mb"],
        model_size=f"{user_factors.size + item_factors.size:,} factors",
        scores=score_all(top, test, args.k),
    )


def build_rows(results: list[Result]) -> tuple[list[str], list[list[str]]]:
    header = ["Metric", *(result.name for result in results)]
    rows = [
        ["Fit time (s)", *(f"{r.fit_seconds:.2f}" for r in results)],
        ["Peak RSS growth (MB)", *(f"{r.rss_growth_mb:.0f}" for r in results)],
        ["Model size", *(r.model_size for r in results)],
    ]
    for name in results[0].scores:
        rows.append([name, *(f"{r.scores[name]:.4f}" for r in results)])
    extra_names = [name for r in results for name in r.extra]
    for name in dict.fromkeys(extra_names):
        rows.append([name, *(r.extra.get(name, "-") for r in results)])
    return header, rows


RSS_FOOTNOTE = (
    "Peak RSS growth is how much the process's peak resident set rose during "
    "that fit. Models are fitted in the order listed, and the peak never falls, "
    "so a later model that needs less memory than an earlier one shows 0."
)


def print_markdown(title: str, results: list[Result]) -> None:
    header, rows = build_rows(results)
    print(f"### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print(f"\n_{RSS_FOOTNOTE}_")


def print_table(title: str, results: list[Result], reporter: Reporter) -> None:
    header, rows = build_rows(results)
    if reporter.console is None:
        label_width = max(len(row[0]) for row in [header, *rows]) + 2
        value_width = max(
            (len(value) for row in [header, *rows] for value in row[1:]), default=0
        )
        print(f"\n{title}")
        for row in [header, *rows]:
            values = "  ".join(value.rjust(value_width) for value in row[1:])
            print(row[0].ljust(label_width) + values)
        return
    table = Table(title=title)
    table.add_column(header[0], style="cyan")
    for name in header[1:]:
        table.add_column(name, justify="right", style="green")
    for row in rows:
        table.add_row(*row)
    reporter.console.print(table)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["100k", "1m", "synthetic"],
        default="100k",
        help="MovieLens variant, or a random matrix that needs no download",
    )
    parser.add_argument("--lambd", type=float, default=2.0, help="SLIM L1 penalty")
    parser.add_argument("--beta", type=float, default=2.0, help="SLIM L2 penalty")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=50,
        help="SLIM passes per item, and ALS iterations",
    )
    parser.add_argument("--tol", type=float, default=1e-6, help="SLIM tolerance")
    parser.add_argument(
        "--threads", type=int, default=None, help="SLIM worker threads (default: all)"
    )
    parser.add_argument("--k", type=int, default=10, help="Cut-off for the metrics")
    parser.add_argument(
        "--baseline",
        choices=["als", "none"],
        default="none",
        help="Also benchmark implicit's ALS",
    )
    parser.add_argument("--factors", type=int, default=64, help="ALS latent factors")
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print a GitHub-flavoured markdown table on stdout",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split and model seed")
    args = parser.parse_args(argv)
    missing = "install the 'bench' dependency group (uv sync --group bench)"
    if args.dataset != "synthetic" and get_movielens is None:
        parser.error(f"--dataset {args.dataset} needs implicit: {missing}")
    if args.baseline == "als" and AlternatingLeastSquares is None:
        parser.error(f"--baseline als needs implicit: {missing}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reporter = Reporter(use_rich=not args.markdown)

    with reporter.step(f"Loading {args.dataset}"):
        matrix = load_dataset(args.dataset, args.seed)
    with reporter.step("Splitting 80/20"):
        train, test = split(matrix, args.seed)

    label = "Synthetic" if args.dataset == "synthetic" else f"MovieLens {args.dataset}"
    title = (
        f"{label} -- {matrix.shape[0]:,} users x {matrix.shape[1]:,} items, "
        f"{matrix.nnz:,} interactions"
    )
    reporter.log(title)
    reporter.log(
        f"lambd={args.lambd} beta={args.beta} max_iter={args.max_iter} "
        f"tol={args.tol} threads={args.threads or 'all'} seed={args.seed}"
    )

    results = [run_slim(train, test, args, reporter)]
    if args.baseline == "als":
        results.append(run_als(train, test, args, reporter))

    if args.markdown:
        print_markdown(title, results)
    else:
        print_table(f"Results (k={args.k})", results, reporter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
