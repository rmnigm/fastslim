# SLIM
Rust implementation of [SLIM (**S**parse **Li**near **M**ethods)](https://ieeexplore.ieee.org/document/6137254) for item-based collaborative filtering with parallel computation.

## Installation Usage

```bash
uv sync
```

## Usage

```python
import slim
from scipy import sparse

# User-item interaction matrix (users x items)
interactions = sparse.random(1000, 500, density=0.01, format='csr')

# Fit model
weights = slim.fit(
    interactions,
    lambda_=0.5,  # L1 regularization (sparsity)
    beta=0.5,     # L2 regularization
)

# Predict scores
scores = slim.predict(interactions, weights)
```

## API

The library exposes one function - `slim.fit`, which fits the model using coordinate descent solver.

**Parameters:**
- `interaction_matrix`: scipy sparse matrix (users x items)
- `lambd`: L1 regularization coefficient
- `beta`: L2 regularization coefficient
- `max_iter`: maximum iterations per item
- `n_threads`: number of threads (None = all cores)

**Returns:** CSR item-item weight matrix

## Algorithm

Optimizes for each item $i$:

$$
L_i = 0.5 * ||a_i - \sum_{j} w_{ij} * a_j||^2 + \lambda * \sum_{j} |w_{ij}| + \beta * \sum_{j} w_{ij}^2
$$

Update rule for non-negative weights:

$$
w_{ik} = \frac{ReLU\left(⟨a_i, a_k⟩ - Σ_{j≠k} w_{ij} * ⟨a_j, a_k⟩ - \lambda \right)}{(||a_k||² + \beta)}
$$

## Tests

```bash
uv sync --extra dev
uv run pytest
```

## Project Structure

```
slim/
├── Cargo.toml
├── pyproject.toml
├── src/
│   └── lib.rs           # Rust solver
└── python/
    └── slim/
        ├── __init__.py  # Python API
        └── test_slim.py # Tests
```
