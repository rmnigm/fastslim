"""Benchmark fastslim SLIM vs implicit ALS on MovieLens dataset."""

from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from typing import Generator

import numpy as np
from scipy import sparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from implicit.datasets.movielens import get_movielens
from implicit.als import AlternatingLeastSquares

import fastslim


@contextmanager
def measure_time() -> Generator[dict[str, float], None, None]:
    result: dict[str, float] = {"elapsed": 0.0}
    start = time.perf_counter()
    yield result
    result["elapsed"] = time.perf_counter() - start


@contextmanager
def measure_memory() -> Generator[dict[str, float], None, None]:
    result: dict[str, float] = {"peak_mb": 0.0}
    tracemalloc.start()
    yield result
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result["peak_mb"] = peak / 1024 / 1024


def load_movielens_data(
    variant: str = "1m",
    seed: int = 42,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    movies, ratings = get_movielens(variant)
    # ratings is (items x users), transpose to (users x items)
    user_item_matrix = ratings.T.tocsr()

    return user_item_matrix.astype(np.float64), movies


def train_test_split(
    matrix: sparse.csr_matrix,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    rng = np.random.default_rng(seed)
    coo = matrix.tocoo()
    mask = rng.random(len(coo.data)) > test_ratio

    train = sparse.csr_matrix(
        (coo.data[mask], (coo.row[mask], coo.col[mask])), shape=matrix.shape
    )
    test = sparse.csr_matrix(
        (coo.data[~mask], (coo.row[~mask], coo.col[~mask])), shape=matrix.shape
    )
    return train, test


def recommend_from_history(
    weights: sparse.csr_matrix,
    history: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """
    Get top-k recommendations from item-item weights and user history.

    Args:
        weights: Item-item weight matrix (items x items)
        history: User interaction vector (items,) - can be binary or weighted
        k: Number of recommendations

    Returns:
        Array of top-k item indices
    """
    scores = history @ weights
    # mask already seen items
    scores[history > 0] = -np.inf
    return np.argpartition(scores, -k)[-k:]


def precision_at_k(
    predictions: np.ndarray, test_matrix: sparse.csr_matrix, k: int = 10
) -> float:
    n_users = predictions.shape[0]
    precisions = []
    test_csr = test_matrix.tocsr()

    for user_idx in range(n_users):
        test_items = set(test_csr[user_idx].indices)
        if not test_items:
            continue
        top_k = np.argpartition(predictions[user_idx], -k)[-k:]
        precisions.append(len(set(top_k) & test_items) / k)

    return np.mean(precisions) if precisions else 0.0


def recall_at_k(
    predictions: np.ndarray, test_matrix: sparse.csr_matrix, k: int = 10
) -> float:
    n_users = predictions.shape[0]
    recalls = []
    test_csr = test_matrix.tocsr()

    for user_idx in range(n_users):
        test_items = set(test_csr[user_idx].indices)
        if not test_items:
            continue
        top_k = np.argpartition(predictions[user_idx], -k)[-k:]
        recalls.append(len(set(top_k) & test_items) / len(test_items))

    return np.mean(recalls) if recalls else 0.0


def ndcg_at_k(
    predictions: np.ndarray, test_matrix: sparse.csr_matrix, k: int = 10
) -> float:
    n_users = predictions.shape[0]
    ndcgs = []
    test_csr = test_matrix.tocsr()

    for user_idx in range(n_users):
        test_items = set(test_csr[user_idx].indices)
        if not test_items:
            continue

        top_k = np.argsort(predictions[user_idx])[::-1][:k]
        dcg = sum(1.0 / np.log2(r + 2) for r, i in enumerate(top_k) if i in test_items)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(test_items), k)))
        if idcg > 0:
            ndcgs.append(dcg / idcg)

    return np.mean(ndcgs) if ndcgs else 0.0


def compute_predictions_batched(
    train: sparse.csr_matrix,
    weights: sparse.csr_matrix,
    batch_size: int = 500,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> np.ndarray:
    n_users = train.shape[0]
    n_items = train.shape[1]
    predictions = np.zeros((n_users, n_items), dtype=np.float32)
    train_csr = train.tocsr()

    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        batch = train_csr[start:end]
        predictions[start:end] = (batch @ weights).toarray()
        # mask seen items
        for i in range(start, end):
            predictions[i, train_csr[i].indices] = -np.inf
        if progress and task_id is not None:
            progress.update(task_id, completed=end)

    return predictions


def benchmark_slim(
    train: sparse.csr_matrix,
    test: sparse.csr_matrix,
    k: int = 10,
    progress: Progress | None = None,
) -> dict:
    with measure_time() as timing, measure_memory() as memory:
        weights = fastslim.fit(train, lambd=0.5, beta=0.5, n_threads=8, max_iter=15)

    task_id = None
    if progress:
        task_id = progress.add_task("SLIM predictions", total=train.shape[0])
    predictions = compute_predictions_batched(
        train, weights, progress=progress, task_id=task_id
    )
    if progress and task_id is not None:
        progress.remove_task(task_id)

    return {
        "time": timing["elapsed"],
        "memory": memory["peak_mb"],
        "precision": precision_at_k(predictions, test, k),
        "recall": recall_at_k(predictions, test, k),
        "ndcg": ndcg_at_k(predictions, test, k),
    }


def compute_als_predictions_batched(
    train: sparse.csr_matrix,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    batch_size: int = 500,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> np.ndarray:
    n_users = train.shape[0]
    n_items = train.shape[1]
    predictions = np.zeros((n_users, n_items), dtype=np.float32)
    train_csr = train.tocsr()

    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        predictions[start:end] = user_factors[start:end] @ item_factors.T
        for i in range(start, end):
            predictions[i, train_csr[i].indices] = -np.inf
        if progress and task_id is not None:
            progress.update(task_id, completed=end)

    return predictions


def benchmark_als(
    train: sparse.csr_matrix,
    test: sparse.csr_matrix,
    k: int = 10,
    progress: Progress | None = None,
) -> dict:
    # implicit expects (users x items) matrix
    model = AlternatingLeastSquares(
        factors=64, regularization=0.01, iterations=15, use_gpu=False
    )

    with measure_time() as timing, measure_memory() as memory:
        model.fit(train.tocsr())

    task_id = None
    if progress:
        task_id = progress.add_task("ALS predictions", total=train.shape[0])
    predictions = compute_als_predictions_batched(
        train,
        model.user_factors,
        model.item_factors,
        progress=progress,
        task_id=task_id,
    )
    if progress and task_id is not None:
        progress.remove_task(task_id)

    return {
        "time": timing["elapsed"],
        "memory": memory["peak_mb"],
        "precision": precision_at_k(predictions, test, k),
        "recall": recall_at_k(predictions, test, k),
        "ndcg": ndcg_at_k(predictions, test, k),
    }


def display_results(slim_res: dict, als_res: dict, console: Console, k: int) -> None:
    table = Table(title=f"Benchmark Results (k={k})")
    table.add_column("Metric", style="cyan")
    table.add_column("SLIM", style="green", justify="right")
    table.add_column("ALS", style="yellow", justify="right")

    table.add_row("Time (s)", f"{slim_res['time']:.2f}", f"{als_res['time']:.2f}")
    table.add_row(
        "Memory (MB)", f"{slim_res['memory']:.1f}", f"{als_res['memory']:.1f}"
    )
    table.add_row(
        f"Precision@{k}", f"{slim_res['precision']:.4f}", f"{als_res['precision']:.4f}"
    )
    table.add_row(
        f"Recall@{k}", f"{slim_res['recall']:.4f}", f"{als_res['recall']:.4f}"
    )
    table.add_row(f"NDCG@{k}", f"{slim_res['ndcg']:.4f}", f"{als_res['ndcg']:.4f}")

    console.print(table)


def main() -> None:
    console = Console()
    k = 10

    console.print(Panel("[bold blue]SLIM vs ALS Benchmark on MovieLens 1M[/bold blue]"))

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Loading MovieLens 1M dataset...", total=None)
        matrix, _ = load_movielens_data("1m")
        progress.remove_task(task)
        console.print(
            f"Dataset: {matrix.shape[0]:,} users, {matrix.shape[1]:,} items, {matrix.nnz:,} interactions"
        )

        task = progress.add_task("Splitting train/test...", total=None)
        train, test = train_test_split(matrix)
        progress.remove_task(task)

        task = progress.add_task("Training ALS...", total=None)
        als_res = benchmark_als(train, test, k, progress=progress)
        progress.remove_task(task)

        task = progress.add_task("Training SLIM...", total=None)
        slim_res = benchmark_slim(train, test, k, progress=progress)
        progress.remove_task(task)

    console.print()
    display_results(slim_res, als_res, console, k)


if __name__ == "__main__":
    main()
