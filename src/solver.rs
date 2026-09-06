//! Per-item non-negative coordinate descent for SLIM (pure Rust, no pyo3).
//!
//! # Objective
//!
//! Let `X` be the (users x items) matrix and `x_i` its `i`-th column. For every
//! target item `i` we solve, independently and in parallel,
//!
//! ```text
//! minimise   0.5 * ||x_i - X w||^2 + lambd * sum_k |w_k| + (beta / 2) * sum_k w_k^2
//! subject to w_k >= 0 for all k,  w_i = 0.
//! ```
//!
//! This is the SLIM objective of Ning & Karypis (2011) with their `beta / 2`
//! convention on the L2 term. With the Gram matrix `P = X^T X` the exact
//! coordinate-wise minimiser is
//!
//! ```text
//! w_k = max(0, (P_ik - sum_{j != k} w_j P_jk - lambd) / (P_kk + beta)).
//! ```
//!
//! # Candidate filter (requires `X >= 0`)
//!
//! Only coordinates with `P_ik >= lambd` are ever considered. This is exact
//! for non-negative data: then `P >= 0` elementwise, so the correction term
//! `sum_{j != k} w_j P_jk` is non-negative and a coordinate with
//! `P_ik < lambd` can never become positive. [`solve_slim_csr`] rejects
//! negative or non-finite data for that reason.
//!
//! # Active set and termination
//!
//! Passes alternate between *full* passes over all candidates and passes over
//! the *active set* fixed after the last full pass. Active-set passes repeat
//! until the largest weight change in a pass is `< tol`; then a full pass
//! runs. If that full pass also moves every coordinate by less than `tol`,
//! the convergence check below runs at the final residuals of the pass and
//! the item is converged if it succeeds. Otherwise the active set is rebuilt
//! (the positive coordinates, plus any the check found wanting to enter) and
//! the cycle repeats.
//!
//! ## Convergence check and guarantee
//!
//! A full pass with `max |delta w| < tol` on its own only bounds the KKT
//! violation of a coordinate by `tol * sum_{j != k} P_jk`, because the
//! coordinates updated later in the pass keep changing its residual. That is
//! a weak bound for a popular item with many correlated neighbours. So after
//! such a pass the solver evaluates, for every candidate `k` and without
//! applying anything, the step the coordinate would take from the *final*
//! residuals `r_k = P_ik - sum_j w_j P_jk`:
//!
//! ```text
//! w_k > 0:  D_k = (r_k - lambd - beta * w_k) / (P_kk + beta)   (unclipped)
//! w_k = 0:  D_k = max(0, r_k - lambd) / (P_kk + beta)          (clipped)
//! ```
//!
//! and the item is converged only if `max_k |D_k| < tol` as well. (For a
//! positive weight the unclipped step is used deliberately: a tiny `w_k`
//! whose update clips to zero would otherwise pass with `|D_k| = w_k` while
//! hiding an arbitrarily large gradient.) The check costs `O(m)` for `m`
//! candidates and is part of the full pass, so it never counts towards
//! `max_iter`. It gives the guarantee that at exit with `converged == true`
//! every candidate `k` satisfies
//!
//! ```text
//! w_k > 0:  |r_k - lambd - beta * w_k| <= (P_kk + beta) * tol
//! w_k = 0:  r_k - lambd               <= (P_kk + beta) * tol
//! ```
//!
//! i.e. its KKT violation is at most `(P_kk + beta) * tol`, up to round-off
//! in the incrementally maintained residuals. Non-candidates
//! (`P_ik < lambd`) satisfy their KKT condition exactly, so the bound holds
//! for every `k != i`.
//!
//! ## Pass budget
//!
//! `max_iter` caps the total number of passes per item, and passes of *both*
//! kinds count (active-set passes as well as full passes), so an
//! ill-conditioned item may consume two to three times the budget that the
//! same number of plain cyclic sweeps would. `tol = 0` disables the stopping
//! test: every item with candidates runs exactly `max_iter` passes and is
//! never flagged converged. When the budget runs out the current iterate is
//! returned as-is. It is always feasible (`w >= 0`, `w_i = 0`) and no pass
//! ever increases the objective, but it is a truncated solution whose KKT
//! violations are not bounded. Such items are reported with
//! `converged == false` and `n_passes == max_iter` in [`CscOutput`]; in
//! particular `max_iter = 0` yields all-zero weights flagged unconverged for
//! every item that has candidates. Items without candidates are trivially
//! converged with `n_passes == 0`.
//!
//! # Numerical guards
//!
//! A Gram matrix with non-finite entries (overflow of `sum_u x_uk x_uj`) is
//! rejected with [`SlimError::InvalidInput`] before solving; it would
//! otherwise turn every update into `NaN` and silently yield an all-zero `W`.
//! A coordinate with `P_kk + beta` not greater than zero (only possible for
//! `beta = 0` when `x_uk^2` underflows) is left at `w_k = 0` instead of
//! dividing by zero, and counts as satisfied in the convergence check.

use crate::gram::Gram;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

/// Solver hyper-parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlimParams {
    /// L1 penalty `lambd >= 0`.
    pub lambd: f64,
    /// L2 penalty `beta >= 0` (objective term is `beta / 2 * ||w||^2`).
    pub beta: f64,
    /// Maximum number of coordinate-descent passes per item. Active-set
    /// passes and full passes both count. An item that exhausts the budget
    /// is returned as it stands and flagged in [`CscOutput::converged`].
    pub max_iter: usize,
    /// Convergence tolerance `tol >= 0` on the largest weight change in a
    /// pass and on the would-be steps of the final check; see the module
    /// docs for the resulting KKT bound. `tol = 0` means exactly `max_iter`
    /// passes per item.
    pub tol: f64,
    /// Rayon thread count; `None` uses the global pool (all cores).
    pub n_threads: Option<usize>,
}

impl Default for SlimParams {
    fn default() -> Self {
        SlimParams {
            lambd: 0.5,
            beta: 0.5,
            max_iter: 1000,
            tol: 1e-4,
            n_threads: None,
        }
    }
}

/// Item-item weights `W` in CSC layout: for target item `i` the segment
/// `indptr[i]..indptr[i+1]` of `indices`/`data` lists neighbours `k`
/// (ascending, never `i` itself) and weights `w_ik > 0`, i.e.
/// `W[k, i] = w_ik` and predictions are `scores = X @ W`.
///
/// `n_passes` and `converged` have one entry per item (see the module docs,
/// "Pass budget").
#[derive(Clone, Debug, PartialEq)]
pub struct CscOutput {
    pub n_items: usize,
    pub indptr: Vec<i64>,
    pub indices: Vec<i64>,
    pub data: Vec<f64>,
    /// Coordinate-descent passes spent on each item, `<= max_iter`.
    pub n_passes: Vec<i64>,
    /// Whether each item passed the convergence check. `false` means it was
    /// cut off at `max_iter` (which includes `max_iter == 0`) and its
    /// weights are a truncated solution.
    pub converged: Vec<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlimError {
    /// Malformed CSR input, out-of-range parameter, or an overflowing Gram.
    InvalidInput(String),
    /// Rayon could not build the requested thread pool.
    ThreadPool(String),
}

impl std::fmt::Display for SlimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlimError::InvalidInput(msg) => write!(f, "{msg}"),
            SlimError::ThreadPool(msg) => write!(f, "failed to build thread pool: {msg}"),
        }
    }
}

impl std::error::Error for SlimError {}

/// Marker for "not a candidate of the current item" in [`Scratch::pos`].
const NOT_CAND: u32 = u32::MAX;

/// Per-thread scratch space reused across items.
struct Scratch {
    /// Global item index -> local candidate index, `NOT_CAND` otherwise.
    pos: Vec<u32>,
    /// Candidate item indices (ascending).
    cand: Vec<u32>,
    /// Residual gradient `r_t = P_ik - sum_j w_j P_jk` for candidate `t`.
    r: Vec<f64>,
    /// Current weights per candidate.
    w: Vec<f64>,
    /// Active-set candidate indices.
    active: Vec<u32>,
}

impl Scratch {
    fn new(n_items: usize) -> Self {
        Scratch {
            pos: vec![NOT_CAND; n_items],
            cand: Vec::new(),
            r: Vec::new(),
            w: Vec::new(),
            active: Vec::new(),
        }
    }
}

/// Solution of one item: neighbours (ascending), weights (`> 0`), passes
/// spent and whether the convergence check succeeded.
struct ItemResult {
    idx: Vec<u32>,
    val: Vec<f64>,
    n_passes: i64,
    converged: bool,
}

/// One coordinate-descent pass over the given local coordinates. Returns the
/// largest absolute weight change.
///
/// Requires `pos.len() == gram.n` and every non-`NOT_CAND` entry of `pos` to
/// be a valid index into `r` and `w` (maintained by [`solve_item`]).
#[inline]
#[allow(clippy::too_many_arguments)]
fn cd_pass<I: Iterator<Item = usize>>(
    coords: I,
    gram: &Gram,
    lambd: f64,
    beta: f64,
    cand: &[u32],
    pos: &[u32],
    r: &mut [f64],
    w: &mut [f64],
) -> f64 {
    debug_assert_eq!(pos.len(), gram.n);
    let mut max_delta = 0.0f64;
    for t in coords {
        let k = cand[t] as usize;
        let diag_k = gram.diag[k];
        let denom = diag_k + beta;
        if denom <= 0.0 {
            // No curvature (beta = 0 and P_kk underflowed): keep w_t = 0.
            continue;
        }
        let c = r[t] + diag_k * w[t];
        let w_new = ((c - lambd) / denom).max(0.0);
        let delta = w_new - w[t];
        if delta != 0.0 {
            w[t] = w_new;
            let (nb_idx, nb_val) = gram.row(k);
            for (&j, &v) in nb_idx.iter().zip(nb_val) {
                // SAFETY: every column index stored in a Gram row is
                // < n_items == pos.len() (inputs are validated before the
                // Gram is built), and any pos entry other than NOT_CAND was
                // set by solve_item to a local index < m == r.len().
                debug_assert!((j as usize) < pos.len());
                let tj = unsafe { *pos.get_unchecked(j as usize) };
                if tj != NOT_CAND {
                    debug_assert!((tj as usize) < r.len());
                    unsafe { *r.get_unchecked_mut(tj as usize) -= v * delta };
                }
            }
            r[t] -= diag_k * delta;
            max_delta = max_delta.max(delta.abs());
        }
    }
    max_delta
}

/// Convergence check at the final residuals of a full pass (module docs,
/// "Convergence check and guarantee"). Returns `max_k |D_k|` without
/// applying anything, and rebuilds `active` as the coordinates that are
/// positive or would become positive.
fn check_pass(
    gram: &Gram,
    lambd: f64,
    beta: f64,
    cand: &[u32],
    r: &[f64],
    w: &[f64],
    active: &mut Vec<u32>,
) -> f64 {
    active.clear();
    let mut max_step = 0.0f64;
    for (t, &k) in cand.iter().enumerate() {
        let denom = gram.diag[k as usize] + beta;
        if denom <= 0.0 {
            continue;
        }
        // Unclipped step, i.e. minus the KKT gradient over the curvature.
        let step = (r[t] - lambd - beta * w[t]) / denom;
        if w[t] > 0.0 {
            active.push(t as u32);
            max_step = max_step.max(step.abs());
        } else if step > 0.0 {
            active.push(t as u32);
            max_step = max_step.max(step);
        }
    }
    max_step
}

/// Solve the per-item problem for target item `i`.
fn solve_item(i: usize, gram: &Gram, p: &SlimParams, s: &mut Scratch) -> ItemResult {
    let Scratch {
        pos,
        cand,
        r,
        w,
        active,
    } = s;
    let lambd = p.lambd;
    let beta = p.beta;

    cand.clear();
    r.clear();
    let (row_idx, row_val) = gram.row(i);
    for (&k, &v) in row_idx.iter().zip(row_val) {
        if v >= lambd {
            cand.push(k);
            r.push(v);
        }
    }
    let m = cand.len();
    if m == 0 || p.max_iter == 0 {
        return ItemResult {
            idx: Vec::new(),
            val: Vec::new(),
            n_passes: 0,
            // Nothing to solve is converged; no budget to solve it is not.
            converged: m == 0,
        };
    }
    for (t, &k) in cand.iter().enumerate() {
        pos[k as usize] = t as u32;
    }
    w.clear();
    w.resize(m, 0.0);
    active.clear();

    let mut full = true;
    let mut passes = 0usize;
    let mut converged = false;
    while passes < p.max_iter {
        passes += 1;
        if full {
            let max_delta = cd_pass(0..m, gram, lambd, beta, cand, pos, r, w);
            if max_delta < p.tol {
                if check_pass(gram, lambd, beta, cand, r, w, active) < p.tol {
                    converged = true;
                    break;
                }
            } else {
                active.clear();
                active.extend((0..m).filter(|&t| w[t] > 0.0).map(|t| t as u32));
            }
            // An empty active set after a moving full pass means everything
            // fell back to zero; let the next full pass decide.
            full = active.is_empty();
        } else {
            let coords = active.iter().map(|&t| t as usize);
            let max_delta = cd_pass(coords, gram, lambd, beta, cand, pos, r, w);
            if max_delta < p.tol {
                full = true;
            }
        }
    }

    let mut idx = Vec::new();
    let mut val = Vec::new();
    for (t, &k) in cand.iter().enumerate() {
        if w[t] > 0.0 {
            idx.push(k);
            val.push(w[t]);
        }
    }
    for &k in cand.iter() {
        pos[k as usize] = NOT_CAND;
    }
    ItemResult {
        idx,
        val,
        n_passes: passes as i64,
        converged,
    }
}

fn assemble(n_items: usize, per_item: Vec<ItemResult>) -> CscOutput {
    let nnz: usize = per_item.iter().map(|item| item.idx.len()).sum();
    let mut indptr = Vec::with_capacity(n_items + 1);
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    let mut n_passes = Vec::with_capacity(n_items);
    let mut converged = Vec::with_capacity(n_items);
    indptr.push(0i64);
    for item in per_item {
        indices.extend(item.idx.iter().map(|&k| k as i64));
        data.extend_from_slice(&item.val);
        indptr.push(indices.len() as i64);
        n_passes.push(item.n_passes);
        converged.push(item.converged);
    }
    CscOutput {
        n_items,
        indptr,
        indices,
        data,
        n_passes,
        converged,
    }
}

fn validate(
    data: &[f64],
    indices: &[u32],
    indptr: &[usize],
    n_rows: usize,
    n_cols: usize,
    params: &SlimParams,
) -> Result<(), SlimError> {
    let err = |msg: String| Err(SlimError::InvalidInput(msg));
    if n_cols >= u32::MAX as usize || n_rows >= u32::MAX as usize {
        return err(format!(
            "matrix too large: n_rows={n_rows}, n_cols={n_cols} must be < {}",
            u32::MAX
        ));
    }
    if indptr.len() != n_rows + 1 {
        return err(format!(
            "indptr has length {}, expected n_rows + 1 = {}",
            indptr.len(),
            n_rows + 1
        ));
    }
    if indptr[0] != 0 {
        return err(format!("indptr[0] must be 0, got {}", indptr[0]));
    }
    if let Some(u) = indptr.windows(2).position(|p| p[1] < p[0]) {
        return err(format!(
            "indptr must be non-decreasing, but indptr[{}]={} > indptr[{}]={}",
            u,
            indptr[u],
            u + 1,
            indptr[u + 1]
        ));
    }
    if indptr[n_rows] != data.len() {
        return err(format!(
            "indptr[n_rows]={} does not match data length {}",
            indptr[n_rows],
            data.len()
        ));
    }
    if indices.len() != data.len() {
        return err(format!(
            "indices length {} does not match data length {}",
            indices.len(),
            data.len()
        ));
    }
    if let Some(bad) = indices.iter().position(|&j| j as usize >= n_cols) {
        return err(format!(
            "column index {} at position {bad} is out of range for n_cols={n_cols}",
            indices[bad]
        ));
    }
    if let Some(bad) = data.iter().position(|v| !(v.is_finite() && *v >= 0.0)) {
        return err(format!(
            "data must be finite and non-negative, got {} at position {bad}",
            data[bad]
        ));
    }
    if params.lambd.is_nan() || params.lambd < 0.0 {
        return err(format!("lambd must be >= 0, got {}", params.lambd));
    }
    if params.beta.is_nan() || params.beta < 0.0 {
        return err(format!("beta must be >= 0, got {}", params.beta));
    }
    if params.tol.is_nan() || params.tol < 0.0 {
        return err(format!("tol must be >= 0, got {}", params.tol));
    }
    if params.n_threads == Some(0) {
        return err("n_threads must be >= 1 or None".to_string());
    }
    Ok(())
}

/// Fit SLIM on a CSR user-item matrix `X` (`n_rows` users x `n_cols` items)
/// and return the item-item weights `W` in CSC layout together with the
/// per-item pass counts and convergence flags (see [`CscOutput`]).
///
/// `data` must be finite and non-negative; duplicate column entries within
/// a row are summed and indices need not be sorted. Values so large that the
/// Gram matrix `X^T X` overflows are rejected. Output is deterministic:
/// identical inputs give byte-identical results regardless of `n_threads`.
pub fn solve_slim_csr(
    data: &[f64],
    indices: &[u32],
    indptr: &[usize],
    n_rows: usize,
    n_cols: usize,
    params: SlimParams,
) -> Result<CscOutput, SlimError> {
    validate(data, indices, indptr, n_rows, n_cols, &params)?;
    let run = || -> Result<CscOutput, SlimError> {
        let gram = Gram::from_csr(data, indices, indptr, n_rows, n_cols);
        if gram.diag.iter().chain(&gram.data).any(|v| !v.is_finite()) {
            return Err(SlimError::InvalidInput(
                "interaction values are too large: Gram matrix overflowed to infinity".to_string(),
            ));
        }
        let per_item: Vec<ItemResult> = (0..n_cols)
            .into_par_iter()
            .map_init(
                || Scratch::new(n_cols),
                |scratch, i| solve_item(i, &gram, &params, scratch),
            )
            .collect();
        Ok(assemble(n_cols, per_item))
    };
    match params.n_threads {
        Some(t) => {
            let pool = ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .map_err(|e| SlimError::ThreadPool(e.to_string()))?;
            pool.install(run)
        }
        None => run(),
    }
}

#[cfg(test)]
mod tests;
