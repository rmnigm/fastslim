//! Python bindings for the SLIM solver. All numerical work lives in
//! [`gram`] and [`solver`]; this module only validates and converts inputs.

mod gram;
mod solver;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use solver::{solve_slim_csr, SlimParams};

type CscArrays<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
);

#[pyfunction]
#[pyo3(signature = (data, indices, indptr, n_rows, n_cols, lambd=0.5, beta=0.5, max_iter=100, tol=1e-6, n_threads=None))]
#[allow(clippy::too_many_arguments)]
fn solve_slim<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, i64>,
    indptr: PyReadonlyArray1<'py, i64>,
    n_rows: usize,
    n_cols: usize,
    lambd: f64,
    beta: f64,
    max_iter: usize,
    tol: f64,
    n_threads: Option<usize>,
) -> PyResult<CscArrays<'py>> {
    let data: Vec<f64> = data.as_slice()?.to_vec();
    let indices: Vec<u32> = indices.as_slice()?.iter().map(|&x| x as u32).collect();
    let indptr: Vec<usize> = indptr.as_slice()?.iter().map(|&x| x as usize).collect();
    let params = SlimParams {
        lambd,
        beta,
        max_iter,
        tol,
        n_threads,
    };
    let out = py
        .allow_threads(|| solve_slim_csr(&data, &indices, &indptr, n_rows, n_cols, params))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        PyArray1::from_vec(py, out.indptr),
        PyArray1::from_vec(py, out.indices),
        PyArray1::from_vec(py, out.data),
    ))
}

#[pymodule]
fn _slim_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_slim, m)?)?;
    Ok(())
}
