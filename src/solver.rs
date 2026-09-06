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
mod tests {
    // The dense reference solver below is written index-by-index on purpose,
    // so that it reads like the formulas in the module docs.
    #![allow(clippy::needless_range_loop)]

    use super::*;
    use crate::test_util::{
        csr_from_dense, near_duplicate_items, random_binary, random_zipf_counts,
    };

    fn dense_gram(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = x[0].len();
        let mut p = vec![vec![0.0; n]; n];
        for row in x {
            for a in 0..n {
                for b in 0..n {
                    p[a][b] += row[a] * row[b];
                }
            }
        }
        p
    }

    /// Dense `W[k][i]` from CSC output.
    fn to_dense_w(out: &CscOutput) -> Vec<Vec<f64>> {
        let n = out.n_items;
        let mut w = vec![vec![0.0; n]; n];
        for i in 0..n {
            for t in out.indptr[i] as usize..out.indptr[i + 1] as usize {
                w[out.indices[t] as usize][i] = out.data[t];
            }
        }
        w
    }

    /// Plain full-pass non-negative coordinate descent over all `k != i`,
    /// run to `max |delta| < 1e-14`. Returns dense `W[k][i]`.
    fn reference(x: &[Vec<f64>], lambd: f64, beta: f64) -> Vec<Vec<f64>> {
        let p = dense_gram(x);
        let n = p.len();
        let mut wmat = vec![vec![0.0; n]; n];
        for i in 0..n {
            let mut w = vec![0.0; n];
            for _ in 0..100_000 {
                let mut max_delta = 0.0f64;
                for k in 0..n {
                    if k == i {
                        continue;
                    }
                    let denom = p[k][k] + beta;
                    if denom == 0.0 {
                        continue;
                    }
                    let mut c = p[i][k];
                    for j in 0..n {
                        if j != k {
                            c -= p[k][j] * w[j];
                        }
                    }
                    let w_new = ((c - lambd) / denom).max(0.0);
                    let d = w_new - w[k];
                    if d != 0.0 {
                        w[k] = w_new;
                        max_delta = max_delta.max(d.abs());
                    }
                }
                if max_delta < 1e-14 {
                    break;
                }
            }
            for k in 0..n {
                wmat[k][i] = w[k];
            }
        }
        wmat
    }

    /// KKT violation of every coordinate `[i][k]` (`k != i`) of `W` for the
    /// objective in the module docs, computed from scratch with the dense
    /// Gram: `|r_k - lambd - beta w_k|` for `w_k > 0`, `max(0, r_k - lambd)`
    /// for `w_k = 0`.
    fn kkt_violations(x: &[Vec<f64>], w: &[Vec<f64>], lambd: f64, beta: f64) -> Vec<Vec<f64>> {
        let p = dense_gram(x);
        let n = p.len();
        let mut viol = vec![vec![0.0; n]; n];
        for i in 0..n {
            for k in 0..n {
                if k == i {
                    continue;
                }
                let mut g = p[i][k];
                for j in 0..n {
                    g -= w[j][i] * p[j][k];
                }
                viol[i][k] = if w[k][i] > 0.0 {
                    (g - lambd - beta * w[k][i]).abs()
                } else {
                    (g - lambd).max(0.0)
                };
            }
        }
        viol
    }

    /// Largest KKT violation of `W`.
    fn kkt_violation(x: &[Vec<f64>], w: &[Vec<f64>], lambd: f64, beta: f64) -> f64 {
        kkt_violations(x, w, lambd, beta)
            .iter()
            .flatten()
            .fold(0.0, |a, &b| f64::max(a, b))
    }

    fn fit(x: &[Vec<f64>], params: SlimParams) -> CscOutput {
        let (data, indices, indptr) = csr_from_dense(x);
        solve_slim_csr(&data, &indices, &indptr, x.len(), x[0].len(), params).unwrap()
    }

    fn tight(lambd: f64, beta: f64) -> SlimParams {
        SlimParams {
            lambd,
            beta,
            max_iter: 10_000,
            tol: 1e-10,
            n_threads: Some(1),
        }
    }

    fn check_invariants(out: &CscOutput) {
        let n = out.n_items;
        assert_eq!(out.n_passes.len(), n);
        assert_eq!(out.converged.len(), n);
        assert!(out.n_passes.iter().all(|&p| p >= 0));
        assert_eq!(out.indptr.len(), n + 1);
        assert_eq!(out.indptr[0], 0);
        assert_eq!(*out.indptr.last().unwrap() as usize, out.indices.len());
        assert_eq!(out.indices.len(), out.data.len());
        assert!(out.indptr.windows(2).all(|p| p[0] <= p[1]));
        for i in 0..n {
            let seg = out.indptr[i] as usize..out.indptr[i + 1] as usize;
            let idx = &out.indices[seg.clone()];
            assert!(idx.windows(2).all(|p| p[0] < p[1]), "segment {i} unsorted");
            assert!(idx.iter().all(|&k| k as usize != i), "self weight in {i}");
            assert!(
                out.data[seg].iter().all(|&w| w > 0.0),
                "non-positive weight"
            );
        }
    }

    fn max_abs_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
        a.iter()
            .zip(b)
            .flat_map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| (x - y).abs()))
            .fold(0.0, f64::max)
    }

    fn nnz(w: &[Vec<f64>]) -> usize {
        w.iter().flatten().filter(|&&v| v > 0.0).count()
    }

    fn assert_matches_reference(x: &[Vec<f64>], lambd: f64, beta: f64) {
        let out = fit(x, tight(lambd, beta));
        check_invariants(&out);
        assert!(out.converged.iter().all(|&c| c), "not all items converged");
        let w = to_dense_w(&out);
        let w_ref = reference(x, lambd, beta);
        assert_eq!(
            nnz(&w),
            nnz(&w_ref),
            "nnz differs (lambd={lambd}, beta={beta})"
        );
        let diff = max_abs_diff(&w, &w_ref);
        assert!(
            diff < 1e-6,
            "max |W - W_ref| = {diff} (lambd={lambd}, beta={beta})"
        );
        let kkt = kkt_violation(x, &w, lambd, beta);
        assert!(
            kkt < 1e-6,
            "KKT violation {kkt} (lambd={lambd}, beta={beta})"
        );
    }

    fn three_items() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 0.0],
        ]
    }

    #[test]
    fn hand_computed_three_items() {
        // P = [[3,2,1],[2,3,2],[1,2,2]], lambd = beta = 0.5.
        // item 0: w_1 = 1.5/3.5 = 3/7, w_2 clipped to 0 (KKT: 0.5 - 2*3/7 < 0)
        // item 1: [3.5 1; 1 2.5] w = [1.5; 1.5]  ->  w = [9/31, 15/31]
        // item 2: w_0 clipped to 0, w_1 = 1.5/3.5 = 3/7
        let out = fit(&three_items(), tight(0.5, 0.5));
        assert_eq!(out.indptr, vec![0, 1, 3, 4]);
        assert_eq!(out.indices, vec![1, 0, 2, 1]);
        let expected = [3.0 / 7.0, 9.0 / 31.0, 15.0 / 31.0, 3.0 / 7.0];
        for (got, want) in out.data.iter().zip(expected) {
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
        assert_matches_reference(&three_items(), 0.5, 0.5);
    }

    #[test]
    fn lambda_zero_matches_reference() {
        assert_matches_reference(&three_items(), 0.0, 0.5);
        assert_matches_reference(&random_binary(21, 40, 10, 0.25), 0.0, 1.0);
    }

    #[test]
    fn beta_zero_matches_reference() {
        assert_matches_reference(&three_items(), 0.5, 0.0);
        assert_matches_reference(&random_binary(22, 40, 10, 0.25), 0.7, 0.0);
    }

    #[test]
    fn huge_lambda_gives_empty() {
        let out = fit(&three_items(), tight(100.0, 0.5));
        check_invariants(&out);
        assert_eq!(out.indptr, vec![0, 0, 0, 0]);
        assert!(out.indices.is_empty() && out.data.is_empty());
        // No candidates: nothing to solve, trivially converged in 0 passes.
        assert_eq!(out.n_passes, vec![0, 0, 0]);
        assert_eq!(out.converged, vec![true, true, true]);
    }

    #[test]
    fn max_iter_zero_gives_empty() {
        let mut p = tight(0.5, 0.5);
        p.max_iter = 0;
        let out = fit(&three_items(), p);
        check_invariants(&out);
        assert_eq!(out.indptr, vec![0, 0, 0, 0]);
        // Every item has candidates (all P_ik >= 0.5) but no budget.
        assert_eq!(out.n_passes, vec![0, 0, 0]);
        assert_eq!(out.converged, vec![false, false, false]);
    }

    #[test]
    fn reports_pass_counts_and_convergence() {
        let x = random_zipf_counts(41, 300, 40);
        let params = |max_iter| SlimParams {
            lambd: 0.5,
            beta: 0.5,
            max_iter,
            tol: 1e-8,
            n_threads: Some(1),
        };

        // Tight budget: the truncated items are flagged and report exactly
        // max_iter passes; the output is still a well-formed feasible point.
        let cut = fit(&x, params(3));
        check_invariants(&cut);
        let n_bad = cut.converged.iter().filter(|&&c| !c).count();
        assert!(n_bad > 0, "expected some items to hit max_iter = 3");
        for i in 0..cut.n_items {
            if cut.converged[i] {
                assert!(cut.n_passes[i] <= 3, "item {i}: {} passes", cut.n_passes[i]);
            } else {
                assert_eq!(cut.n_passes[i], 3, "item {i}");
            }
        }

        // Generous budget: everything converges within it, and items that
        // ended up with weights needed at least one pass.
        let ok = fit(&x, params(10_000));
        check_invariants(&ok);
        assert!(ok.converged.iter().all(|&c| c));
        for i in 0..ok.n_items {
            assert!(ok.n_passes[i] <= 10_000);
            if ok.indptr[i + 1] > ok.indptr[i] {
                assert!(ok.n_passes[i] >= 1, "item {i}");
            }
        }
        assert!(
            ok.n_passes.iter().any(|&p| p > 3),
            "budget of 3 was not binding"
        );

        // tol = 0 disables the stopping test: exactly max_iter passes, never
        // flagged converged (for items with candidates, i.e. all of them here).
        let exact = fit(
            &x,
            SlimParams {
                tol: 0.0,
                ..params(7)
            },
        );
        assert!(exact.n_passes.iter().all(|&p| p == 7));
        assert!(exact.converged.iter().all(|&c| !c));
    }

    #[test]
    fn converged_items_meet_kkt_bound() {
        // At exit with converged == true every coordinate's KKT violation is
        // at most (P_kk + beta) * tol (module docs, "Convergence check and
        // guarantee"). Loose tolerances make the bound, not closeness to
        // the optimum, the property under test. On the random problems a
        // plain `max |delta w| < tol` stop already satisfies it (violations
        // reach ~0.75 of the bound); on the near-duplicate problems, where
        // coordinate descent crawls along `sum_k w_k` and the coordinates of
        // a full pass all move together, that stop overshoots the bound by
        // 1.3x-4.5x and only the final check brings it to <= 1.0. The
        // relative slack covers round-off in the incrementally maintained
        // residuals.
        let problems = [
            random_binary(31, 60, 20, 0.3),
            random_zipf_counts(32, 400, 50),
            near_duplicate_items(34, 500, 12, 20),
            near_duplicate_items(35, 2000, 20, 10),
        ];
        for x in &problems {
            let p = dense_gram(x);
            for (lambd, beta) in [(0.5, 0.5), (0.1, 0.0), (2.0, 1.0)] {
                for tol in [1e-2, 1e-4, 1e-6] {
                    let out = fit(
                        x,
                        SlimParams {
                            lambd,
                            beta,
                            max_iter: 1_000_000,
                            tol,
                            n_threads: Some(1),
                        },
                    );
                    check_invariants(&out);
                    assert!(out.converged.iter().all(|&c| c), "tol={tol}");
                    let viol = kkt_violations(x, &to_dense_w(&out), lambd, beta);
                    for i in 0..out.n_items {
                        for k in 0..out.n_items {
                            let bound = (p[k][k] + beta) * tol;
                            assert!(
                                viol[i][k] <= bound * (1.0 + 1e-6),
                                "item {i}, coord {k}: violation {} > bound {bound} \
                                 (lambd={lambd}, beta={beta}, tol={tol})",
                                viol[i][k]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_gram_overflow() {
        // 1e200^2 overflows P to +inf; without the guard every update turns
        // into NaN and W comes back silently empty.
        let (mut data, indices, indptr) = csr_from_dense(&three_items());
        for v in data.iter_mut() {
            *v = 1e200;
        }
        let err =
            solve_slim_csr(&data, &indices, &indptr, 4, 3, SlimParams::default()).unwrap_err();
        assert!(
            matches!(&err, SlimError::InvalidInput(msg) if msg.contains("overflow")),
            "{err}"
        );
        // Just below the threshold the same matrix solves fine.
        for v in data.iter_mut() {
            *v = 1e150;
        }
        let out = solve_slim_csr(&data, &indices, &indptr, 4, 3, SlimParams::default()).unwrap();
        assert!(out.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn zero_curvature_coordinate_is_left_at_zero() {
        // One user with x = [1e150, 1e-170]: P_00 = 1e300 and P_01 = 1e-20
        // are finite but P_11 = 1e-340 underflows to 0. With beta = 0 the
        // step for w_1 of item 0 would divide by zero; it must be skipped.
        let x = vec![vec![1e150, 1e-170]];
        let out = fit(
            &x,
            SlimParams {
                lambd: 0.0,
                beta: 0.0,
                max_iter: 100,
                tol: 1e-12,
                n_threads: Some(1),
            },
        );
        check_invariants(&out);
        assert!(out.data.iter().all(|v| v.is_finite()));
        assert_eq!(out.indptr[0], out.indptr[1], "item 0 must have no weights");
        assert!(out.converged.iter().all(|&c| c));
    }

    #[test]
    fn item_with_small_norm_is_not_skipped() {
        // P_00 = 3 < lambd^2 = 4, yet P_01 = 3 >= lambd, so w_10 = 1/10.5 > 0.
        let mut x = vec![vec![0.0; 3]; 10];
        for (u, row) in x.iter_mut().enumerate() {
            if u < 3 {
                row[0] = 1.0;
            }
            row[1] = 1.0;
            if u >= 5 {
                row[2] = 1.0;
            }
        }
        let out = fit(&x, tight(2.0, 0.5));
        let w = to_dense_w(&out);
        assert!((w[1][0] - 1.0 / 10.5).abs() < 1e-9, "w_10 = {}", w[1][0]);
        assert_matches_reference(&x, 2.0, 0.5);
    }

    #[test]
    fn kkt_on_random_binary_problems() {
        for seed in 1..=3u64 {
            let x = random_binary(seed, 40, 15, 0.2);
            for lambd in [0.1, 0.5, 1.5] {
                assert_matches_reference(&x, lambd, 0.5);
            }
        }
    }

    #[test]
    fn deterministic_across_thread_counts() {
        let x = random_binary(5, 300, 60, 0.1);
        let run = |n_threads| {
            fit(
                &x,
                SlimParams {
                    lambd: 0.5,
                    beta: 0.5,
                    max_iter: 100,
                    tol: 1e-6,
                    n_threads,
                },
            )
        };
        let one = run(Some(1));
        assert!(!one.data.is_empty());
        assert_eq!(one, run(Some(1)));
        assert_eq!(one, run(Some(4)));
        assert_eq!(one, run(None));
    }

    #[test]
    fn output_invariants_on_random_problem() {
        let x = random_binary(9, 200, 50, 0.15);
        for lambd in [0.0, 0.5, 2.0] {
            let out = fit(&x, tight(lambd, 0.5));
            check_invariants(&out);
        }
    }

    #[test]
    fn duplicates_are_summed_like_canonical_input() {
        // Replace the first stored entry of row 0 (value 1.0) by two entries
        // 0.75 + 0.25 for the same column; the result must not change.
        let x = random_binary(13, 20, 6, 0.4);
        let (data, indices, indptr) = csr_from_dense(&x);
        assert!(indptr[1] > 0, "test assumes row 0 is non-empty");
        let params = tight(0.3, 0.5);
        let canonical = solve_slim_csr(&data, &indices, &indptr, 20, 6, params).unwrap();
        let mut d2 = vec![0.75, 0.25];
        d2.extend_from_slice(&data[1..]);
        let mut i2 = vec![indices[0], indices[0]];
        i2.extend_from_slice(&indices[1..]);
        let p2: Vec<usize> = indptr
            .iter()
            .enumerate()
            .map(|(u, &p)| if u == 0 { 0 } else { p + 1 })
            .collect();
        let dup = solve_slim_csr(&d2, &i2, &p2, 20, 6, params).unwrap();
        assert_eq!(dup.indptr, canonical.indptr);
        assert_eq!(dup.indices, canonical.indices);
        assert!(max_abs_diff(&to_dense_w(&dup), &to_dense_w(&canonical)) < 1e-12);
    }

    #[test]
    fn rejects_invalid_input() {
        let (data, indices, indptr) = csr_from_dense(&three_items());
        let ok = SlimParams::default();
        let bad = |d: &[f64], ix: &[u32], ip: &[usize], p: SlimParams| {
            matches!(
                solve_slim_csr(d, ix, ip, 4, 3, p),
                Err(SlimError::InvalidInput(_))
            )
        };
        assert!(solve_slim_csr(&data, &indices, &indptr, 4, 3, ok).is_ok());
        assert!(bad(&data, &indices, &indptr[..4], ok)); // indptr too short
        assert!(bad(&data, &indices, &[1, 2, 5, 7, 8], ok)); // indptr[0] != 0
        assert!(bad(&data, &indices, &[0, 5, 2, 7, 8], ok)); // non-monotone
        assert!(bad(&data, &indices, &[0, 2, 5, 7, 9], ok)); // indptr[n] != nnz
        assert!(bad(&data, &indices[..7], &indptr, ok)); // indices/data mismatch
        let mut oob = indices.to_vec();
        oob[0] = 3;
        assert!(bad(&data, &oob, &indptr, ok)); // index out of range
        let mut nan = data.to_vec();
        nan[1] = f64::NAN;
        assert!(bad(&nan, &indices, &indptr, ok));
        let mut neg = data.to_vec();
        neg[1] = -1.0;
        assert!(bad(&neg, &indices, &indptr, ok));
        for p in [
            SlimParams { lambd: -1.0, ..ok },
            SlimParams { beta: -1.0, ..ok },
            SlimParams { tol: -1.0, ..ok },
            SlimParams {
                lambd: f64::NAN,
                ..ok
            },
            SlimParams {
                n_threads: Some(0),
                ..ok
            },
        ] {
            assert!(bad(&data, &indices, &indptr, p), "{p:?} accepted");
        }
    }
}
