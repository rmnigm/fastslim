//! Gram matrix `P = X^T X` of the user-item matrix in sparse form.

use rayon::prelude::*;
use std::collections::HashMap;

const EPS: f64 = 0.0;

pub struct Gram {
    pub diag: Vec<f64>,
    pub nbrs: Vec<Vec<(usize, f64)>>,
}

impl Gram {
    pub fn from_csr(
        data: &[f64],
        indices: &[usize],
        indptr: &[usize],
        n_rows: usize,
        n_cols: usize,
    ) -> Self {
        let mut diag = vec![0.0; n_cols];
        for row in 0..n_rows {
            let start = indptr[row];
            let end = indptr[row + 1];
            for idx in start..end {
                let col = indices[idx];
                let val = data[idx];
                diag[col] = val.mul_add(val, diag[col]);
            }
        }

        let pairs = (0..n_rows)
            .into_par_iter()
            .fold(HashMap::new, |mut local, row| {
                let start = indptr[row];
                let end = indptr[row + 1];
                let row_indices = &indices[start..end];
                let row_data = &data[start..end];
                let len = row_indices.len();
                for a in 0..len {
                    let ia = row_indices[a];
                    let va = row_data[a];
                    for b in (a + 1)..len {
                        let ib = row_indices[b];
                        let vb = row_data[b];
                        let (lo, hi) = if ia < ib { (ia, ib) } else { (ib, ia) };
                        let key = ((lo as u64) << 32) | (hi as u64);
                        let entry = local.entry(key).or_insert(0.0);
                        *entry = va.mul_add(vb, *entry);
                    }
                }
                local
            })
            .reduce(HashMap::new, |mut acc, local| {
                for (key, val) in local {
                    let entry = acc.entry(key).or_insert(0.0);
                    *entry += val;
                }
                acc
            });

        let mut nbrs: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_cols];
        for (key, val) in pairs {
            if val.abs() <= EPS {
                continue;
            }
            let a = (key >> 32) as usize;
            let b = (key & 0xFFFF_FFFF) as usize;
            nbrs[a].push((b, val));
            nbrs[b].push((a, val));
        }
        Self { diag, nbrs }
    }
}
