//! Gram matrix `P = X^T X` of the user-item matrix `X`, in sparse form.
//!
//! The fixed-order sparse accumulation that makes this deterministic is
//! described in `docs/algorithm.md`.

use rayon::prelude::*;

use crate::scratch::PerThread;

/// Sparse symmetric Gram matrix `P = X^T X`: off-diagonals row by row in CSR
/// with ascending indices and no stored zeros, the diagonal split into `diag`.
#[derive(Clone, Debug, PartialEq)]
pub struct Gram {
    pub n: usize,
    pub diag: Vec<f64>,
    pub indptr: Vec<usize>,
    pub indices: Vec<u32>,
    pub data: Vec<f64>,
}

/// Compressed-sparse-column view of `X`: `row_idx[col_ptr[k]..col_ptr[k+1]]`
/// are the users (ascending) who consumed item `k`, with matching `vals`.
struct Csc {
    col_ptr: Vec<usize>,
    row_idx: Vec<u32>,
    vals: Vec<f64>,
}

fn transpose_csr(
    data: &[f64],
    indices: &[u32],
    indptr: &[usize],
    n_rows: usize,
    n_cols: usize,
) -> Csc {
    let nnz = data.len();
    let mut col_ptr = vec![0usize; n_cols + 1];
    for &j in indices {
        col_ptr[j as usize + 1] += 1;
    }
    for k in 0..n_cols {
        col_ptr[k + 1] += col_ptr[k];
    }
    let mut next = col_ptr[..n_cols].to_vec();
    let mut row_idx = vec![0u32; nnz];
    let mut vals = vec![0.0f64; nnz];
    for u in 0..n_rows {
        for idx in indptr[u]..indptr[u + 1] {
            let j = indices[idx] as usize;
            let dst = next[j];
            next[j] += 1;
            row_idx[dst] = u as u32;
            vals[dst] = data[idx];
        }
    }
    Csc {
        col_ptr,
        row_idx,
        vals,
    }
}

/// Per-thread scratch for the sparse accumulator.
struct Spa {
    acc: Vec<f64>,
    mark: Vec<bool>,
    touched: Vec<u32>,
}

impl Spa {
    fn new(n_cols: usize) -> Self {
        Spa {
            acc: vec![0.0; n_cols],
            mark: vec![false; n_cols],
            touched: Vec::new(),
        }
    }
}

/// Accumulate row `k` of `P` into `spa.acc`, recording the touched columns.
fn accumulate_row(
    k: usize,
    csc: &Csc,
    data: &[f64],
    indices: &[u32],
    indptr: &[usize],
    spa: &mut Spa,
) {
    let Spa { acc, mark, touched } = spa;
    touched.clear();
    for idx in csc.col_ptr[k]..csc.col_ptr[k + 1] {
        let u = csc.row_idx[idx] as usize;
        let x_uk = csc.vals[idx];
        let row = indptr[u]..indptr[u + 1];
        for (&j, &x_uj) in indices[row.clone()].iter().zip(&data[row]) {
            let j = j as usize;
            if !mark[j] {
                mark[j] = true;
                touched.push(j as u32);
            }
            acc[j] += x_uk * x_uj;
        }
    }
}

/// Number of structurally nonzero off-diagonal entries of row `k`, an upper
/// bound on what [`emit_row`] stores; leaves `spa` cleared.
fn count_row(k: usize, csc: &Csc, indices: &[u32], indptr: &[usize], spa: &mut Spa) -> usize {
    let Spa { mark, touched, .. } = spa;
    touched.clear();
    for &u in &csc.row_idx[csc.col_ptr[k]..csc.col_ptr[k + 1]] {
        let u = u as usize;
        for &j in &indices[indptr[u]..indptr[u + 1]] {
            let j = j as usize;
            if !mark[j] {
                mark[j] = true;
                touched.push(j as u32);
            }
        }
    }
    let count = touched.len() - usize::from(mark[k]);
    for &j in touched.iter() {
        mark[j as usize] = false;
    }
    count
}

/// Write the nonzero off-diagonals of row `k` (ascending) to the front of
/// `out_idx`/`out_val` and return `(P_kk, entries written)`; leaves `spa` cleared.
fn emit_row(k: usize, spa: &mut Spa, out_idx: &mut [u32], out_val: &mut [f64]) -> (f64, usize) {
    let Spa { acc, mark, touched } = spa;
    touched.sort_unstable();
    let mut diag_k = 0.0;
    let mut t = 0;
    for &j in touched.iter() {
        let ju = j as usize;
        let v = acc[ju];
        if ju == k {
            diag_k = v;
        } else if v != 0.0 {
            out_idx[t] = j;
            out_val[t] = v;
            t += 1;
        }
        acc[ju] = 0.0;
        mark[ju] = false;
    }
    (diag_k, t)
}

/// Split `buf` into consecutive mutable segments delimited by `indptr`.
fn split_rows<'a, T>(mut buf: &'a mut [T], indptr: &[usize]) -> Vec<&'a mut [T]> {
    indptr
        .windows(2)
        .map(|w| {
            let (head, tail) = std::mem::take(&mut buf).split_at_mut(w[1] - w[0]);
            buf = tail;
            head
        })
        .collect()
}

impl Gram {
    /// Build `P = X^T X` from a validated CSR matrix, summing duplicate column
    /// entries within a row the way scipy does.
    pub fn from_csr(
        data: &[f64],
        indices: &[u32],
        indptr: &[usize],
        n_rows: usize,
        n_cols: usize,
    ) -> Self {
        debug_assert_eq!(indptr.len(), n_rows + 1);
        debug_assert_eq!(indices.len(), data.len());
        // User and item indices are stored as u32.
        debug_assert!(n_rows < u32::MAX as usize && n_cols < u32::MAX as usize);

        let csc = transpose_csr(data, indices, indptr, n_rows, n_cols);
        let spas = PerThread::new(|| Spa::new(n_cols));

        // Pass 1 bounds every row so that pass 2 can write into one allocation.
        let bounds: Vec<usize> = (0..n_cols)
            .into_par_iter()
            .map(|k| spas.with(|spa| count_row(k, &csc, indices, indptr, spa)))
            .collect();
        let mut indptr_out = Vec::with_capacity(n_cols + 1);
        indptr_out.push(0);
        let mut bound_nnz = 0;
        for b in bounds {
            bound_nnz += b;
            indptr_out.push(bound_nnz);
        }

        let mut indices_out = vec![0u32; bound_nnz];
        let mut data_out = vec![0.0f64; bound_nnz];
        let rows: Vec<(f64, usize)> = split_rows(&mut indices_out, &indptr_out)
            .into_par_iter()
            .zip(split_rows(&mut data_out, &indptr_out))
            .enumerate()
            .map(|(k, (out_idx, out_val))| {
                spas.with(|spa| {
                    accumulate_row(k, &csc, data, indices, indptr, spa);
                    emit_row(k, spa, out_idx, out_val)
                })
            })
            .collect();

        // Close the gaps left by sums that came out exactly zero.
        if rows.iter().map(|&(_, len)| len).sum::<usize>() < bound_nnz {
            let (mut nnz, mut start) = (0, 0);
            for (k, &(_, len)) in rows.iter().enumerate() {
                let next = indptr_out[k + 1];
                indices_out.copy_within(start..start + len, nnz);
                data_out.copy_within(start..start + len, nnz);
                nnz += len;
                indptr_out[k + 1] = nnz;
                start = next;
            }
            indices_out.truncate(nnz);
            data_out.truncate(nnz);
        }
        let diag = rows.into_iter().map(|(d, _)| d).collect();

        Gram {
            n: n_cols,
            diag,
            indptr: indptr_out,
            indices: indices_out,
            data: data_out,
        }
    }

    /// Off-diagonal entries of row `k`: `(indices, values)`, indices ascending.
    #[inline]
    pub fn row(&self, k: usize) -> (&[u32], &[f64]) {
        let (s, e) = (self.indptr[k], self.indptr[k + 1]);
        (&self.indices[s..e], &self.data[s..e])
    }
}

#[cfg(test)]
mod tests;
