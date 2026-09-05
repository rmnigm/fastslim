//! Per-item non-negative coordinate descent for SLIM (pure Rust, no pyo3).

use crate::gram::Gram;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::collections::HashMap;

const ACTIVE_WARM_ITER: usize = 2;
const TOL: f64 = 1e-6;

/// Solver hyper-parameters.
#[derive(Clone, Copy, Debug)]
pub struct SlimParams {
    pub lambd: f64,
    pub beta: f64,
    pub max_iter: usize,
    pub n_threads: Option<usize>,
}

/// Item-item weights `W` in CSC layout: for target item `i` the segment
/// `indptr[i]..indptr[i+1]` of `indices`/`data` lists neighbours `k` and
/// weights `w_ik > 0`, i.e. `W[k, i] = w_ik` and `scores = X @ W`.
#[derive(Clone, Debug, PartialEq)]
pub struct CscOutput {
    pub n_items: usize,
    pub indptr: Vec<i64>,
    pub indices: Vec<i64>,
    pub data: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlimError {
    ThreadPool(String),
}

impl std::fmt::Display for SlimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlimError::ThreadPool(msg) => write!(f, "failed to build thread pool: {msg}"),
        }
    }
}

impl std::error::Error for SlimError {}

fn solve_item(i: usize, gram: &Gram, lambd: f64, beta: f64, max_iter: usize) -> Vec<(usize, f64)> {
    if gram.diag[i] < lambd * lambd {
        return vec![];
    }
    if gram.nbrs[i].is_empty() {
        return vec![];
    }
    let mut cand = Vec::new();
    let mut r = Vec::new();
    let mut pos: HashMap<usize, usize> = HashMap::new();
    for (k, val) in gram.nbrs[i].iter() {
        if *val < lambd {
            continue;
        }
        let t = cand.len();
        cand.push(*k);
        r.push(*val);
        pos.insert(*k, t);
    }
    let m = cand.len();
    if m == 0 {
        return vec![];
    }
    let mut w = vec![0.0; m];
    let mut active: Vec<usize> = (0..m).collect();
    for iter in 0..max_iter {
        let mut max_abs_delta = 0.0;
        let mut next_active = Vec::new();
        for &t in active.iter() {
            let k = cand[t];
            let diag_k = gram.diag[k];
            let a = diag_k + beta;
            let c = r[t] + diag_k * w[t];
            let w_new = ((c - lambd) / a).max(0.0);
            let delta = w_new - w[t];
            if delta != 0.0 {
                w[t] = w_new;
                for (j, val) in gram.nbrs[k].iter() {
                    if let Some(&pos_j) = pos.get(j) {
                        r[pos_j] -= val * delta;
                    }
                }
                r[t] -= diag_k * delta;
                let abs_delta = delta.abs();
                if abs_delta > max_abs_delta {
                    max_abs_delta = abs_delta;
                }
            }
            if iter >= ACTIVE_WARM_ITER && (w_new != 0.0 || c > lambd) {
                next_active.push(t);
            }
        }
        if iter >= ACTIVE_WARM_ITER {
            if next_active.is_empty() {
                break;
            }
            active = next_active;
        }
        if max_abs_delta < TOL {
            break;
        }
    }
    w.into_iter()
        .enumerate()
        .filter_map(|(t, val)| {
            if val > 0.0 {
                Some((cand[t], val))
            } else {
                None
            }
        })
        .collect()
}

fn assemble(n_items: usize, per_item: Vec<Vec<(usize, f64)>>) -> CscOutput {
    let nnz: usize = per_item.iter().map(Vec::len).sum();
    let mut indptr = Vec::with_capacity(n_items + 1);
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    indptr.push(0);
    for item in per_item {
        for (k, w) in item {
            indices.push(k as i64);
            data.push(w);
        }
        indptr.push(indices.len() as i64);
    }
    CscOutput {
        n_items,
        indptr,
        indices,
        data,
    }
}

/// Fit SLIM on a CSR user-item matrix and return the item-item weights.
pub fn solve_slim_csr(
    data: &[f64],
    indices: &[usize],
    indptr: &[usize],
    n_rows: usize,
    n_cols: usize,
    params: SlimParams,
) -> Result<CscOutput, SlimError> {
    let run = || {
        let gram = Gram::from_csr(data, indices, indptr, n_rows, n_cols);
        let per_item: Vec<_> = (0..n_cols)
            .into_par_iter()
            .map(|i| solve_item(i, &gram, params.lambd, params.beta, params.max_iter))
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
