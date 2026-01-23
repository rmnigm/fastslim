# fastslim
Rust implementation of [SLIM (**S**parse **Li**near **M**ethods)](https://ieeexplore.ieee.org/document/6137254) for item-based collaborative filtering with parallel computation.

## Installation

```bash
uv sync
uv run maturin develop
```

## Usage

```python
import fastslim
from scipy import sparse

# User-item interaction matrix (users x items)
interactions = sparse.random(1000, 500, density=0.01, format='csr')

# Fit model
weights = fastslim.fit(
    interactions,
    lambd=0.5,  # L1 regularization (sparsity)
    beta=0.5,     # L2 regularization
)
```

## API

The library exposes one function - `fastslim.fit`, which fits the model using coordinate descent solver.

**Parameters:**
- `interaction_matrix`: scipy sparse matrix (users x items)
- `lambd`: L1 regularization coefficient
- `beta`: L2 regularization coefficient
- `max_iter`: maximum iterations per item
- `n_threads`: number of threads (None = all cores)

**Returns:** CSR item-item weight matrix

## Algorithm

This section is about math behind the model, some optimization of coordinate descend for the specific scenario with non-negative weights and zero diagonal in item-to-item matrix.

### Objective (per item)

We solve each item independently with coordinate descent:

$$
L_i = \frac{1}{2} \left\| a_i - \sum_j w_{ij} a_j \right\|^2 + \lambda \sum_j |w_{ij}| + \beta \sum_j w_{ij}^2 \rightarrow \min_{w_{i1}\dots w_{in}}
$$

Since we solve for non-negative weights, the absolute value drops and the update is projected onto the feasible set.

The coordinate-wise optimum (gradient = 0) is:

$$
w_{ik} = \frac{\langle a_i, a_k \rangle - \sum_{j \ne k} w_{ij} \langle a_j, a_k \rangle - \lambda}{\|a_k\|^2 + \beta}
$$

Projected to non-negative weights:

$$
w_{ik} = \max\left(0, \frac{\langle a_i, a_k \rangle - \sum_{j \ne k} w_{ij} \langle a_j, a_k \rangle - \lambda}{\|a_k\|^2 + \beta} \right)
$$

### Practical speedups (why they work)

These follow directly from the non-negativity constraint and the structure of $P = X^T X$:

- If $\langle a_i, a_k \rangle < \lambda$, then even with all other weights zero the update is non-positive, so $w_{ik}$ will always be 0. We skip such coordinates.
- If for item $i$ the interaction norm is too small ($P_{ii} < \lambda^2$), the entire solution is zero and we can skip the item.

### Implementation (step-by-step)

1) `Gram::from_csr(...)` - build the Gram matrix $P = X^T X$ in sparse form (from CSR rows).
   - Diagonal: `P_kk = sum(value^2)` per item.
   - Off-diagonal: for each user row, accumulate all item pairs `(a, b)` with `P_ab += v_a * v_b`.
   - Convert pair map into symmetric adjacency lists `nbrs[k] = {(j, P_kj)}`.

2) `solve_item(...)` - for each target item `i`, solve only over neighbors of `i`.
   - Candidates are `cand = { k | P_ik != 0 }`, filtered by `P_ik >= lambd` (valid under non-negative inputs).
   - Initialize residuals `r[t] = P_ik` for each local candidate index `t`.

3) `solve_item(...)` - coordinate descent with residual maintenance (non-negative weights).
   - For each candidate `k`, update:
     - `a = P_kk + beta`
     - `c = r[t] + P_kk * w[t]`
     - `w_new = max(0, (c - lambd) / a)`
   - If `delta = w_new - w_old` is non-zero, update residuals:
     - For each neighbor `(j, P_kj)` in `nbrs[k]` that is a candidate:
       `r[pos(j)] -= P_kj * delta`
     - Also update self residual: `r[t] -= P_kk * delta`.

4) `solve_item(...)` - active set and early stopping of optimizer
   - After a short warm-up, iterate only candidates that are non-zero or still potentially active (`c > lambd`).
   - Stop if max weight change falls below a tolerance, or after `max_iter`.

5) `solve_slim(...)` - ssemble sparse weights.
   - Return only `(k, w_ki)` where `w > 0`.
   - No self-edges, because diagonal is always zero.

## Tests

```bash
uv sync --extra dev
uv run pytest
```
