use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::collections::HashMap;

const EPS: f64 = 0.0;
const ACTIVE_WARM_ITER: usize = 2;
const TOL: f64 = 1e-6;

struct Gram {
    diag: Vec<f64>,
    nbrs: Vec<Vec<(usize, f64)>>,
}

impl Gram {
    fn from_csr(
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

#[pyclass]
pub struct SlimResult {
    #[pyo3(get)]
    pub rows: Vec<usize>,
    #[pyo3(get)]
    pub cols: Vec<usize>,
    #[pyo3(get)]
    pub data: Vec<f64>,
    #[pyo3(get)]
    pub shape: (usize, usize),
}

#[pyfunction]
#[pyo3(signature = (data, indices, indptr, n_rows, n_cols, lambd=0.5, beta=0.5, max_iter=100, n_threads=None))]
#[allow(clippy::too_many_arguments)]
fn solve_slim(
    py: Python<'_>,
    data: PyReadonlyArray1<f64>,
    indices: PyReadonlyArray1<i64>,
    indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    lambd: f64,
    beta: f64,
    max_iter: usize,
    n_threads: Option<usize>,
) -> PyResult<SlimResult> {
    let data: Vec<f64> = data.as_slice()?.to_vec();
    let indices: Vec<usize> = indices.as_slice()?.iter().map(|&x| x as usize).collect();
    let indptr: Vec<usize> = indptr.as_slice()?.iter().map(|&x| x as usize).collect();
    let pool = if let Some(t) = n_threads {
        Some(
            ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .map_err(|err| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string())
                })?,
        )
    } else {
        None
    };
    let (rows, cols, weights) = py.allow_threads(|| {
        let run = || {
            let gram = Gram::from_csr(&data, &indices, &indptr, n_rows, n_cols);
            let results: Vec<_> = (0..n_cols)
                .into_par_iter()
                .map(|i| solve_item(i, &gram, lambd, beta, max_iter))
                .collect();
            let mut rows = vec![];
            let mut cols = vec![];
            let mut weights = vec![];
            for (col, item_weights) in results.into_iter().enumerate() {
                for (row, w) in item_weights {
                    rows.push(row);
                    cols.push(col);
                    weights.push(w);
                }
            }
            (rows, cols, weights)
        };
        if let Some(ref pool) = pool {
            pool.install(run)
        } else {
            run()
        }
    });
    Ok(SlimResult {
        rows,
        cols,
        data: weights,
        shape: (n_cols, n_cols),
    })
}

#[pymodule]
fn _slim_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_slim, m)?)?;
    m.add_class::<SlimResult>()?;
    Ok(())
}
