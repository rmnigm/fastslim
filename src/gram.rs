//! Gram matrix `P = X^T X` of the user-item matrix `X`, in sparse form.
//!
//! `X` arrives as CSR (users x items). We build a CSC view (items -> users)
//! and then, independently for every item `k`, accumulate row `k` of `P` with
//! a dense sparse-accumulator: for each user `u` who consumed `k`, walk row
//! `u` and add `x_uk * x_uj` into `acc[j]`. Every row of `P` is therefore
//! produced by a fixed sequential summation order, so the result is
//! bit-identical regardless of thread count or scheduling.

use rayon::prelude::*;

/// Sparse symmetric Gram matrix `P = X^T X`.
///
/// Off-diagonal entries are stored row by row (CSR layout) with column
/// indices sorted ascending. The diagonal `P_kk` is kept in `diag` and never
/// appears in a row, so row `k` never contains `k` itself. Diagonal and
/// off-diagonal values come from the same accumulator, so duplicate entries
/// within a row of `X` are summed consistently. Exact zeros (from explicitly
/// stored zeros in `X`) are not stored.
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

/// Row `k` of `P`: `(P_kk, off-diagonal indices ascending, values)`.
fn gram_row(
    k: usize,
    csc: &Csc,
    data: &[f64],
    indices: &[u32],
    indptr: &[usize],
    spa: &mut Spa,
) -> (f64, Vec<u32>, Vec<f64>) {
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
    touched.sort_unstable();
    let mut diag_k = 0.0;
    let mut out_idx = Vec::with_capacity(touched.len());
    let mut out_val = Vec::with_capacity(touched.len());
    for &j in touched.iter() {
        let ju = j as usize;
        let v = acc[ju];
        if ju == k {
            diag_k = v;
        } else if v != 0.0 {
            out_idx.push(j);
            out_val.push(v);
        }
        acc[ju] = 0.0;
        mark[ju] = false;
    }
    (diag_k, out_idx, out_val)
}

impl Gram {
    /// Build `P = X^T X` from a validated CSR matrix (`indices` in
    /// `[0, n_cols)`, `indptr` monotone with `indptr[n_rows] == data.len()`).
    /// Duplicate column entries within a row are summed like scipy does.
    pub fn from_csr(
        data: &[f64],
        indices: &[u32],
        indptr: &[usize],
        n_rows: usize,
        n_cols: usize,
    ) -> Self {
        debug_assert_eq!(indptr.len(), n_rows + 1);
        debug_assert_eq!(indices.len(), data.len());

        let csc = transpose_csr(data, indices, indptr, n_rows, n_cols);

        let rows: Vec<(f64, Vec<u32>, Vec<f64>)> = (0..n_cols)
            .into_par_iter()
            .map_init(
                || Spa::new(n_cols),
                |spa, k| gram_row(k, &csc, data, indices, indptr, spa),
            )
            .collect();

        let nnz: usize = rows.iter().map(|(_, idx, _)| idx.len()).sum();
        let diag: Vec<f64> = rows.iter().map(|(d, _, _)| *d).collect();
        let mut indptr_out = Vec::with_capacity(n_cols + 1);
        let mut indices_out = Vec::with_capacity(nnz);
        let mut data_out = Vec::with_capacity(nnz);
        indptr_out.push(0);
        for (_, idx, val) in &rows {
            indices_out.extend_from_slice(idx);
            data_out.extend_from_slice(val);
            indptr_out.push(indices_out.len());
        }

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
