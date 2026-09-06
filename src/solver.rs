//! Per-item non-negative coordinate descent for SLIM (pure Rust, no pyo3).
//!
//! The objective, the candidate restriction, the active set, the convergence
//! check and its KKT bound, the pass budget and the numerical guards are all
//! derived in `docs/algorithm.md`.

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
    /// Maximum coordinate-descent passes per item; both kinds of pass count.
    pub max_iter: usize,
    /// Convergence tolerance `tol >= 0`; `tol = 0` means exactly `max_iter` passes.
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

/// Item-item weights `W` in CSC layout: segment `i` lists the neighbours `k` of
/// target item `i` (ascending) with `W[k, i] > 0`, so `scores = X @ W`.
#[derive(Clone, Debug, PartialEq)]
pub struct CscOutput {
    pub n_items: usize,
    pub indptr: Vec<i64>,
    pub indices: Vec<i64>,
    pub data: Vec<f64>,
    /// Coordinate-descent passes spent on each item, `<= max_iter`.
    pub n_passes: Vec<i64>,
    /// Whether each item passed the convergence check; `false` means truncated.
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

/// One coordinate-descent pass over `coords`, returning the largest weight change.
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
                // SAFETY: Gram column indices are < pos.len() and every pos
                // entry other than NOT_CAND is a local index < r.len().
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

/// Convergence check at the final residuals of a full pass: return `max_k |D_k|`
/// without applying anything, and rebuild `active`.
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
            // An empty active set means everything fell back to zero.
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

/// Fit SLIM on a CSR user-item matrix and return `W` in CSC layout with the
/// per-item pass counts and convergence flags.
///
/// `data` must be finite and non-negative. The output is byte-identical for
/// identical inputs regardless of `n_threads`.
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
