use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

fn get_column(
    data: &[f64],
    indices: &[usize],
    indptr: &[usize],
    n_rows: usize,
    col: usize,
) -> Vec<(usize, f64)> {
    (0..n_rows)
        .filter_map(|row| {
            let row_indices = &indices[indptr[row]..indptr[row + 1]];
            row_indices
                .binary_search(&col)
                .ok()
                .map(|pos| (row, data[indptr[row] + pos]))
        })
        .collect()
}

fn sparse_dot(a: &[(usize, f64)], b: &[(usize, f64)]) -> f64 {
    let (mut i, mut j, mut dot) = (0, 0, 0.0);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                dot = a[i].1.mul_add(b[j].1, dot);
                i += 1;
                j += 1;
            }
        }
    }
    dot
}

struct PrecomputedDotProduct(HashMap<(usize, usize), f64>);

impl PrecomputedDotProduct {
    fn new(data: &[f64], indices: &[usize], indptr: &[usize], n_rows: usize, n_cols: usize) -> Self {
        let cols: Vec<_> = (0..n_cols)
            .into_par_iter()
            .map(|c| get_column(data, indices, indptr, n_rows, c))
            .collect();

        let mut dots = HashMap::new();
        for j in 0..n_cols {
            for k in j..n_cols {
                let d = sparse_dot(&cols[j], &cols[k]);
                if d != 0.0 {
                    dots.insert((j, k), d);
                    if j != k {
                        dots.insert((k, j), d);
                    }
                }
            }
        }

        Self(dots)
    }

    fn dot(&self, j: usize, k: usize) -> f64 {
        self.0.get(&(j, k)).copied().unwrap_or(0.0)
    }

    fn norm(&self, j: usize) -> f64 {
        self.dot(j, j)
    }
}

fn solve_item(i: usize, p: &PrecomputedDotProduct, lambda: f64, beta: f64, n: usize, max_iter: usize) -> Vec<(usize, f64)> {
    if p.norm(i) < lambda * lambda {
        return vec![];
    }
    let mut w = vec![0.0; n];
    for _ in 0..max_iter {
        for k in 0..n {
            let norm_k = p.norm(k);
            if k == i || norm_k == 0.0 {
                continue;
            }
            let dot_ik = p.dot(i, k);
            if dot_ik < lambda {
                w[k] = 0.0;
                continue;
            }
            let sum: f64 = (0..n)
                .filter(|&j| j != k && j != i && w[j] != 0.0)
                .map(|j| w[j] * p.dot(j, k))
                .sum();
            w[k] = ((dot_ik - sum - lambda) / (norm_k + beta)).max(0.0);
        }
    }

    w.into_iter()
        .enumerate()
        .filter(|&(_, v)| v > 0.0)
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

    if let Some(t) = n_threads {
        rayon::ThreadPoolBuilder::new().num_threads(t).build_global().ok();
    }

    let (rows, cols, weights) = py.allow_threads(|| {
        let p = PrecomputedDotProduct::new(&data, &indices, &indptr, n_rows, n_cols);

        let results: Vec<_> = (0..n_cols)
            .into_par_iter()
            .map(|i| solve_item(i, &p, lambd, beta, n_cols, max_iter))
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
    });

    Ok(SlimResult { rows, cols, data: weights, shape: (n_cols, n_cols) })
}

#[pymodule]
fn _slim_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_slim, m)?)?;
    m.add_class::<SlimResult>()?;
    Ok(())
}
