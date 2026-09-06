from importlib import metadata

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
