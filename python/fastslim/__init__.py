"""fastslim -- SLIM (Sparse Linear Methods) item-item recommender in Rust.

>>> import numpy as np
>>> from scipy import sparse
>>> import fastslim
>>> X = sparse.csr_matrix(np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]]))
>>> W = fastslim.fit(X, lambd=0.1, beta=0.1)
>>> fastslim.recommend(W, X[0], k=1)
array([2])
"""

from importlib import metadata as metadata

from . import metrics
from .api import ConvergenceWarning, fit, predict, recommend
from .estimator import SLIM, NotFittedError

__all__ = [
    "SLIM",
    "ConvergenceWarning",
    "NotFittedError",
    "__version__",
    "fit",
    "metrics",
    "predict",
    "recommend",
]

try:
    __version__ = metadata.version("fastslim")
except metadata.PackageNotFoundError:  # pragma: no cover - source tree fallback
    __version__ = "0.0.0+unknown"
