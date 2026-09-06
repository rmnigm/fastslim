# The fastslim solver

How `fastslim.fit` turns a user-item matrix into item-item weights, and why each
shortcut it takes is exact rather than approximate. The code that implements this lives
in [`src/gram.rs`](../src/gram.rs) (the Gram matrix) and
[`src/solver.rs`](../src/solver.rs) (coordinate descent); the Python side in
[`python/fastslim/_api.py`](../python/fastslim/_api.py) only validates and converts.

## 1. Notation and the objective

$X \in \mathbb{R}^{m \times n}$ is the user-item interaction matrix: $m$ users,
$n$ items, entries finite and **non-negative**. $x_i$ is its $i$-th column, the
interaction vector of item $i$ across all users.

SLIM learns an item-item matrix $W \in \mathbb{R}^{n \times n}$ such that
$X W \approx X$, with $W$ sparse, non-negative and zero on the diagonal (otherwise
each item would trivially reconstruct itself). Because the Frobenius objective and both
penalties are separable over the columns of $W$, the problem splits into $n$
independent problems, one per item. Writing $w = W_{:,i}$ for the $i$-th column:

$$
\min_{w}\;
f_i(w) = \tfrac{1}{2}\,\lVert x_i - X w \rVert_2^2
\;+\; \lambda \sum_{k} w_k
\;+\; \tfrac{\beta}{2} \sum_{k} w_k^2
\qquad \text{s.t.}\quad w \ge 0,\; w_i = 0 .
$$

`lambd` is $\lambda \ge 0$, `beta` is $\beta \ge 0$.

Two consequences of $w \ge 0$ that the rest of this document leans on:

* $\lVert w \rVert_1 = \sum_k w_k$, so the $\ell_1$ term is *linear*, not just
  piecewise linear. $f_i$ is a smooth quadratic on the feasible set — no subgradients
  needed anywhere.
* $f_i$ is strictly convex whenever $\beta > 0$, so the minimiser is unique. At
  $\beta = 0$ it is convex but possibly flat along the null space of $X$, and the
  minimiser need not be unique.

## 2. Everything is a function of the Gram matrix

Let $P = X^\top X$, so $P_{ik} = \langle x_i, x_k \rangle$. Expanding the squared
norm:

$$
\tfrac{1}{2}\lVert x_i - X w \rVert^2
= \tfrac{1}{2} x_i^\top x_i - w^\top X^\top x_i + \tfrac{1}{2} w^\top X^\top X w
= \tfrac{1}{2} P_{ii} - \sum_k w_k P_{ik} + \tfrac{1}{2} \sum_{k,j} w_k w_j P_{kj},
$$

so

$$
f_i(w) = \tfrac{1}{2} P_{ii}
- \sum_k w_k P_{ik}
+ \tfrac{1}{2} \sum_{k,j} w_k w_j P_{kj}
+ \lambda \sum_k w_k
+ \tfrac{\beta}{2} \sum_k w_k^2 .
$$

$X$ appears nowhere except through $P$. The solver therefore touches the interaction
data exactly once, to build $P$, and then works entirely in item space. Two properties
of $P$ matter later:

* $P$ is symmetric positive semi-definite (it is a Gram matrix), so $P_{kk} \ge 0$.
* Because $X \ge 0$ entrywise, $P \ge 0$ entrywise. This is the property that makes
  the candidate restriction in §4 exact, and it is why
  [`solve_slim_csr`](../src/solver.rs) rejects negative input outright rather than
  silently returning an approximation.

## 3. The coordinate-wise minimiser

Hold every $w_j$, $j \ne k$, fixed and read $f_i$ as a function of $w_k$ alone:

$$
f_i(w_k) = \tfrac{1}{2}\,(P_{kk} + \beta)\, w_k^2
- \bigl( \underbrace{P_{ik} - \textstyle\sum_{j \ne k} w_j P_{jk}}_{\textstyle c_k} - \lambda \bigr) w_k
+ \text{const}.
$$

That is an upward parabola with vertex at $(c_k - \lambda) / (P_{kk} + \beta)$.
Minimising it over the half-line $w_k \ge 0$ is just projection onto that half-line,
because a convex function of one variable is monotone on either side of its vertex:

$$
\boxed{\;w_k \;\leftarrow\; \max\!\left(0,\; \frac{c_k - \lambda}{P_{kk} + \beta}\right),
\qquad c_k = P_{ik} - \sum_{j \ne k} w_j P_{jk}. \;}
$$

This is the non-negative special case of the usual soft-threshold update for the elastic
net: with the sign constraint, $\operatorname{soft}_\lambda(\cdot)$ collapses to
$\max(0, \cdot - \lambda)$.

**The denominator is never zero.** $P_{kk} + \beta = 0$ requires both $\beta = 0$ and
$P_{kk} = 0$, i.e. item $k$ has no interactions at all. But such an item has
$P_{ik} = 0$ for every $i$, and §4 shows those coordinates are never even considered.
Conversely, any coordinate the solver does consider has $P_{ik} \neq 0$, which forces
some user to have consumed both $i$ and $k$, hence $P_{kk} > 0$. No guard is needed in
the inner loop.

## 4. Candidate restriction, and why it is exact

The solver never allocates a full length-$n$ weight vector. For target item $i$ it
considers only

$$
C_i = \{\, k : P_{ik} \text{ is a stored (nonzero) off-diagonal entry and } P_{ik} \ge \lambda \,\},
$$

which in the code is the filter `if v >= lambd` over row $i$ of the Gram matrix. Every
excluded coordinate is provably zero at the optimum — this is a restriction of the
search space, not a heuristic.

**Claim.** If $X \ge 0$ and $k \notin C_i$, then $w_k = 0$ at every minimiser of
$f_i$, and the algorithm never assigns it a positive value.

*Proof.* Take any feasible $w \ge 0$. Since $P \ge 0$ entrywise,
$\sum_{j \ne k} w_j P_{jk} \ge 0$, hence

$$
c_k = P_{ik} - \sum_{j \ne k} w_j P_{jk} \;\le\; P_{ik}.
$$

Two cases cover everything outside $C_i$:

* **$P_{ik} < \lambda$.** Then $c_k - \lambda \le P_{ik} - \lambda < 0$, so the
  coordinate-wise minimiser of §3 is exactly $0$, whatever the other weights are. The
  KKT condition for a positive weight (§7) reads $c_k = \lambda + \beta w_k \ge \lambda$
  and cannot hold either, so no minimiser has $w_k > 0$.
* **$P_{ik} = 0$** (item $k$ was never co-consumed with $i$; entries equal to zero are
  not stored in the Gram matrix at all). Then $c_k \le 0 \le \lambda$, so
  $\max(0, (c_k - \lambda)/(P_{kk}+\beta)) = 0$ again. This case matters when
  $\lambda = 0$, where the inequality $P_{ik} < \lambda$ is unavailable. $\square$

Note the filter is `>=`, not `>`. A coordinate with $P_{ik} = \lambda$ exactly is also
provably zero (the same argument gives $c_k - \lambda \le 0$), so `>` would be exact
too; `>=` simply keeps one harmless extra candidate rather than reasoning about an
equality in floating point.

The diagonal constraint $w_i = 0$ needs no code at all: `Gram` stores $P_{kk}$ in a
separate `diag` array and never puts $k$ into row $k$, so $i \notin C_i$ by
construction.

> The 0.1.x releases also skipped an entire item when $P_{ii} < \lambda^2$. That test
> was not valid — it does not follow from anything above — and it has been removed.

## 5. Residual maintenance

Recomputing $c_k = P_{ik} - \sum_{j \ne k} w_j P_{jk}$ from scratch would cost a full
pass over row $k$ of $P$ for *every* coordinate visit, including the vast majority that
change nothing. Instead the solver keeps, for each candidate $t$ (holding item $k$), the
**full** residual

$$
r_t \;=\; P_{ik} - \sum_{j} w_j P_{jk}
$$

where the sum now runs over *all* candidates including $j = k$. The partial residual is
recovered in one multiply-add:

$$
c_k = r_t + P_{kk}\, w_k ,
$$

which is exactly the line `let c = r[t] + diag_k * w[t];`. Initialising $w = 0$ makes
$r_t = P_{ik}$, which is what `solve_item` pushes while it builds the candidate list.

When a coordinate moves by $\delta = w_k^{\text{new}} - w_k$, the residual of every
other candidate $j$ changes by exactly $-\delta P_{jk}$, and $r_t$ itself by
$-\delta P_{kk}$. The update therefore walks row $k$ of $P$ once, and only when
$\delta \ne 0$:

```text
for (j, P_jk) in row(k):
    if j is a candidate:  r[pos(j)] -= P_jk * delta
r[t] -= P_kk * delta
```

`pos` is a dense length-$n$ lookup from item index to local candidate index, reused
across items and reset on the way out, so the membership test is a single array read
rather than a hash lookup or a binary search. The residual invariant holds exactly (up
to floating point) at all times, which is what lets the stopping rule in §7 say
something about KKT violations.

This is the "covariance update" form of coordinate descent from Friedman, Hastie and
Tibshirani's glmnet: cost is proportional to the number of *accepted* updates times the
Gram row length, not to the number of coordinate visits.

## 6. The active set and the stopping rule

Most coordinates are zero at the optimum and stay zero. Sweeping all of $C_i$ every
pass wastes work, but sweeping only the currently positive ones can never bring a new
coordinate in. `solve_item` alternates:

* A **full pass** sweeps every candidate `0..m`, then rebuilds the active set
  $A = \{\, t : w_t > 0 \,\}$.
* **Active-set passes** sweep only $A$, repeatedly, until one of them moves nothing by
  as much as `tol`. Then the next pass is a full one again.
* The item is **converged** when a *full* pass moves no coordinate by `tol` or more.
* If a full pass moves something but leaves $A$ empty, the next pass is full as well —
  there is nothing to iterate over otherwise.

`max_iter` caps the **total** number of passes, full and active-set alike. Two edge
cases fall out of this directly:

* `max_iter=0` returns an all-zero $W$: no pass ever runs.
* `tol=0` never satisfies `max_delta < tol` (not even at `max_delta == 0.0`), so the
  solver performs exactly `max_iter` passes for every item.

An item that runs out of `max_iter` before a full pass comes back quiet returns a
*truncated* solution: a feasible, non-negative $W$ column that has not reached the
optimum. `fit` reports this through `fastslim.ConvergenceWarning`, and `SLIM` records it
per item in `n_passes_` and `converged_`.

> 0.1.x removed a coordinate from the active set permanently once it hit zero, so a
> weight that should have re-entered later never could. That is why 0.2.0 results differ
> from 0.1.x even at identical hyperparameters: the old answer was not the optimum.

## 7. What convergence guarantees

The KKT conditions for $\min f_i(w)$ over $w \ge 0$ use the gradient of the smooth
objective,

$$
\frac{\partial f_i}{\partial w_k} = -\Bigl(P_{ik} - \sum_j w_j P_{jk}\Bigr) + \lambda + \beta w_k
= -\,r_t + \lambda + \beta w_k ,
$$

giving, for every $k \ne i$,

$$
w_k > 0 \;\Longrightarrow\; r_t - \lambda - \beta w_k = 0,
\qquad
w_k = 0 \;\Longrightarrow\; r_t - \lambda \le 0 .
$$

Define the **violation** of coordinate $k$ as $\lvert r_t - \lambda - \beta w_k \rvert$
when $w_k > 0$ and $\max(0,\, r_t - \lambda)$ when $w_k = 0$. An exact solution scores
zero. (This is precisely what `kkt_violation` in
[`tests/conftest.py`](../tests/conftest.py) computes, independently of the Rust code.)

Now take the returned weights, with their exact residuals, and evaluate the update of §3
once more without applying it; call the step it would take $\delta_k$.

* **$w_k = 0$.** Here $c_k = r_t$, so the violation is $\max(0, c_k - \lambda)$. If
  $c_k \le \lambda$ that is zero and $\delta_k = 0$. Otherwise
  $\delta_k = (c_k - \lambda)/(P_{kk} + \beta)$ and the violation is exactly
  $(P_{kk} + \beta)\,\lvert \delta_k \rvert$.
* **$w_k > 0$ and $c_k > \lambda$.** Substituting
  $r_t = c_k - P_{kk} w_k$ and $(P_{kk}+\beta)\,w_k^{\text{new}} = c_k - \lambda$,

  $$
  r_t - \lambda - \beta w_k = (c_k - \lambda) - (P_{kk} + \beta) w_k
  = (P_{kk} + \beta)\,(w_k^{\text{new}} - w_k) = (P_{kk} + \beta)\,\delta_k ,
  $$

  so the violation is again exactly $(P_{kk} + \beta)\,\lvert \delta_k \rvert$.
* **$k \notin C_i$.** $r_t \le P_{ik} < \lambda$ (or $\le 0 = \lambda$), so the
  violation is zero — §4.

A `converged` exit means a full pass over all of $C_i$ in which no $\lvert \delta
\rvert$ reached `tol`. Combined with the identity above, **every candidate's KKT
violation at the returned solution is at most $(P_{kk} + \beta) \cdot \texttt{tol}$**.
The bound scales with $P_{kk}$, an item's popularity, which is the formal version of
the practical advice in the README: on count-scale data where $P_{kk}$ reaches the
thousands, a fixed `tol` buys much less accuracy than it does on small data, and `beta`
is the knob that fixes the conditioning.

(The one case the identity does not cover is a positive weight that a fresh update would
push back to zero. That requires $\lvert \delta_k \rvert = w_k < \texttt{tol}$ — a
weight smaller than the tolerance itself — and zeroing it restores exact optimality for
that coordinate.)

The test suite checks this end to end rather than trusting the derivation: at
`tol=1e-10` the worst observed violation over a grid of $\lambda \in [0, 3]$ and
$\beta \in \{0, 0.5, 2\}$, on both binary and weighted matrices, is about `2e-9`, and
the solution matches a naive dense reference solver entrywise.

## 8. Building the Gram matrix, deterministically

$X$ arrives as CSR (users x items). `Gram::from_csr` first transposes it into a CSC view
(item -> the users who consumed it), then computes each row $k$ of $P$ independently
with a **dense sparse accumulator**: for every user $u$ who consumed $k$, walk user
$u$'s CSR row and add $x_{uk} x_{uj}$ into `acc[j]`. The list of touched columns is
sorted once at the end, the diagonal $P_{kk}$ is split off into `diag`, and exact zeros
are dropped.

Floating-point addition is not associative, so "deterministic" is a claim about
summation *order*, and three things secure it:

1. **A row's sum is never split.** Row $k$ is accumulated by one thread, in one
   sequential loop, in an order fixed by the data: users ascending (from the CSC
   transpose, which fills column by column in row order) and, within a user, columns
   ascending (the input is canonicalised to sorted CSR before it reaches Rust). Rayon
   parallelises *across* rows, so it changes which thread does a sum, never the order
   inside one.
2. **No hash maps.** The accumulator is a plain `Vec<f64>` indexed by item, with a
   `Vec<bool>` mark array and an explicit `touched` list that is sorted before emission.
   There is no iteration over a hash table whose order depends on capacity, seed or
   insertion history.
3. **One item, one thread, start to finish.** `solve_item` owns a target item's entire
   solve; `assemble` concatenates the per-item results in item order. Rayon's
   `map_init(...).collect()` fills a `Vec` by index, so the output layout is independent
   of completion order.

Together these give the property the tests assert byte for byte: `fastslim.fit(X,
n_threads=1)` and `fastslim.fit(X, n_threads=None)` produce identical `indptr`,
`indices` and `data` buffers.

Duplicate entries within a row of $X$ are summed rather than dropped, and the diagonal
comes out of the *same* accumulator as the off-diagonal entries, so $P_{kk}$ can never
disagree with the rest of row $k$. (0.1.x computed the diagonal separately and got a
nonzero $W$ diagonal on inputs with duplicate CSR entries.)

## 9. Complexity and memory

Let $\text{nnz}(X)$ be the number of interactions, $|u|$ the number of items in user
$u$'s row, and $\text{nnz}(P)$ the number of co-occurring item pairs.

| Phase | Time | Extra memory |
| --- | --- | --- |
| CSC transpose | $O(\text{nnz}(X) + n)$ | $O(\text{nnz}(X) + n)$ |
| Gram accumulation | $O\!\left(\sum_u \lvert u \rvert^2\right)$ plus $O(\text{nnz}(P)\log n)$ for the per-row sorts | $O(n)$ per thread for the accumulator |
| Coordinate descent | `max_iter` x $O(\text{nnz}(P))$ worst case; in practice proportional to the number of *accepted* updates times the Gram row length | $O(n)$ per thread for `pos`, plus $O(\lvert C_i \rvert)$ for the working vectors |

The Gram matrix dominates steady-state memory at
$12 \cdot \text{nnz}(P) + 8n$ bytes (a `u32` index and an `f64` value per stored entry,
plus the diagonal), and it is built once and shared immutably across threads. The
$\sum_u \lvert u \rvert^2$ term is why power users with enormous histories are the
expensive part of a fit, not the item count.

The output has $\text{nnz}(W) \le \text{nnz}(P)$ and, in practice, far less: on
MovieLens 1M with `lambd=2.0` the Gram matrix has millions of pairs while $W$ keeps
210,095 weights.

## 10. Differences from the paper

Against Ning & Karypis, *SLIM: Sparse Linear Methods for Top-N Recommender Systems*
(ICDM 2011):

* **Same objective, including the $\beta/2$ convention.** The paper's regulariser is
  $\tfrac{\beta}{2}\lVert W \rVert_F^2 + \lambda \lVert W \rVert_1$, and that is what
  the solver implements. (The 0.1.x README documented $\beta \lVert w \rVert^2$; the
  code was right and the README was wrong.)
* **Non-negativity is exploited, not just imposed.** The paper states $W \ge 0$ as a
  constraint. Here it additionally licenses the exact candidate restriction of §4, which
  is what makes each per-item problem small.
* **No feature-selection approximation.** The paper's `fsSLIM` variant speeds fits up by
  pre-selecting a top-$N$ neighbourhood per item with a similarity measure, which
  changes the answer. `fastslim` restricts candidates only where the restriction is
  provably lossless, so the result is the solution of the full problem.
* **Per-item independence is the parallelism axis.** The paper notes that the columns of
  $W$ decouple; here that is exactly the unit of work handed to rayon, and the reason
  the answer does not depend on the thread count.
* **One fit, not a path.** There is no regularisation path, no intercept and no
  cross-validation built in; a fit is a single $(\lambda, \beta)$ pair.

## References

* X. Ning and G. Karypis, "SLIM: Sparse Linear Methods for Top-N Recommender Systems",
  *ICDM 2011*. <https://ieeexplore.ieee.org/document/6137254>
* J. Friedman, T. Hastie and R. Tibshirani, "Regularization Paths for Generalized Linear
  Models via Coordinate Descent", *Journal of Statistical Software* 33(1), 2010 — the
  active set and covariance-update scheme of §5 and §6.
