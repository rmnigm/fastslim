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
//! the *active set* `{k : w_k > 0}` fixed after the last full pass. Active-set
//! passes repeat until the largest weight change in a pass is `< tol`; then a
//! full pass runs, and if it too moves every coordinate by less than `tol`
//! the item is converged. Otherwise the active set is rebuilt and the cycle
//! repeats. `max_iter` caps the total number of passes (full and active-set
//! passes both count); `max_iter = 0` yields all-zero weights.

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
    /// Maximum number of coordinate-descent passes per item.
    pub max_iter: usize,
    /// Convergence tolerance on `max |delta w|` within a pass, `tol >= 0`.
    pub tol: f64,
    /// Rayon thread count; `None` uses the global pool (all cores).
    pub n_threads: Option<usize>,
}

impl Default for SlimParams {
    fn default() -> Self {
        SlimParams {
            lambd: 0.5,
            beta: 0.5,
            max_iter: 100,
            tol: 1e-6,
            n_threads: None,
        }
    }
}

/// Item-item weights `W` in CSC layout: for target item `i` the segment
/// `indptr[i]..indptr[i+1]` of `indices`/`data` lists neighbours `k`
/// (ascending, never `i` itself) and weights `w_ik > 0`, i.e.
/// `W[k, i] = w_ik` and predictions are `scores = X @ W`.
#[derive(Clone, Debug, PartialEq)]
pub struct CscOutput {
    pub n_items: usize,
    pub indptr: Vec<i64>,
    pub indices: Vec<i64>,
    pub data: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlimError {
    /// Malformed CSR input or out-of-range parameter.
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

/// One coordinate-descent pass over the given local coordinates. Returns the
/// largest absolute weight change.
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
    let mut max_delta = 0.0f64;
    for t in coords {
        let k = cand[t] as usize;
        let diag_k = gram.diag[k];
        let c = r[t] + diag_k * w[t];
        let w_new = ((c - lambd) / (diag_k + beta)).max(0.0);
        let delta = w_new - w[t];
        if delta != 0.0 {
            w[t] = w_new;
            let (nb_idx, nb_val) = gram.row(k);
            for (&j, &v) in nb_idx.iter().zip(nb_val) {
                let tj = pos[j as usize];
                if tj != NOT_CAND {
                    r[tj as usize] -= v * delta;
                }
            }
            r[t] -= diag_k * delta;
            max_delta = max_delta.max(delta.abs());
        }
    }
    max_delta
}

/// Solve the per-item problem for target item `i`. Returns `(neighbours,
/// weights)` with neighbours ascending and all weights `> 0`.
fn solve_item(i: usize, gram: &Gram, p: &SlimParams, s: &mut Scratch) -> (Vec<u32>, Vec<f64>) {
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
        return (Vec::new(), Vec::new());
    }
    for (t, &k) in cand.iter().enumerate() {
        pos[k as usize] = t as u32;
    }
    w.clear();
    w.resize(m, 0.0);
    active.clear();

    let mut full = true;
    let mut passes = 0;
    while passes < p.max_iter {
        passes += 1;
        if full {
            let max_delta = cd_pass(0..m, gram, lambd, beta, cand, pos, r, w);
            active.clear();
            active.extend((0..m).filter(|&t| w[t] > 0.0).map(|t| t as u32));
            if max_delta < p.tol {
                break;
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

    let mut out_idx = Vec::new();
    let mut out_val = Vec::new();
    for (t, &k) in cand.iter().enumerate() {
        if w[t] > 0.0 {
            out_idx.push(k);
            out_val.push(w[t]);
        }
    }
    for &k in cand.iter() {
        pos[k as usize] = NOT_CAND;
    }
    (out_idx, out_val)
}

fn assemble(n_items: usize, per_item: Vec<(Vec<u32>, Vec<f64>)>) -> CscOutput {
    let nnz: usize = per_item.iter().map(|(idx, _)| idx.len()).sum();
    let mut indptr = Vec::with_capacity(n_items + 1);
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    indptr.push(0i64);
    for (idx, val) in per_item {
        indices.extend(idx.iter().map(|&k| k as i64));
        data.extend_from_slice(&val);
        indptr.push(indices.len() as i64);
    }
    CscOutput {
        n_items,
        indptr,
        indices,
        data,
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
/// and return the item-item weights `W` in CSC layout (see [`CscOutput`]).
///
/// `data` must be finite and non-negative; duplicate column entries within
/// a row are summed and indices need not be sorted. Output is
/// deterministic: identical inputs give byte-identical results regardless of
/// `n_threads`.
pub fn solve_slim_csr(
    data: &[f64],
    indices: &[u32],
    indptr: &[usize],
    n_rows: usize,
    n_cols: usize,
    params: SlimParams,
) -> Result<CscOutput, SlimError> {
    validate(data, indices, indptr, n_rows, n_cols, &params)?;
    let run = || {
        let gram = Gram::from_csr(data, indices, indptr, n_rows, n_cols);
        let per_item: Vec<(Vec<u32>, Vec<f64>)> = (0..n_cols)
            .into_par_iter()
            .map_init(
                || Scratch::new(n_cols),
                |scratch, i| solve_item(i, &gram, &params, scratch),
            )
            .collect();
        assemble(n_cols, per_item)
    };
    match params.n_threads {
        Some(t) => {
            let pool = ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .map_err(|e| SlimError::ThreadPool(e.to_string()))?;
            Ok(pool.install(run))
        }
        None => Ok(run()),
    }
}
