//! Python bindings for the SLIM solver. All numerical work lives in
//! [`gram`] and [`solver`]; this module only validates and converts inputs.

mod gram;
mod solver;
#[cfg(test)]
mod test_util;

use std::borrow::Cow;

use numpy::{Element, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use solver::{solve_slim_csr, SlimError, SlimParams};

type CscArrays<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
);

/// scipy stores CSR index arrays as int32 or int64; accept both as-is.
#[derive(FromPyObject)]
enum IndexArray<'py> {
    I32(PyReadonlyArray1<'py, i32>),
    I64(PyReadonlyArray1<'py, i64>),
}

/// Convert an integer index array to `Vec<U>`, rejecting values that do not
/// fit (negative, or beyond the target range) with a `ValueError`.
fn convert_indices<T, U>(arr: &PyReadonlyArray1<'_, T>, name: &str) -> PyResult<Vec<U>>
where
    T: Element + Copy + TryInto<U> + std::fmt::Display,
{
    let view = arr.as_array();
    let mut out = Vec::with_capacity(view.len());
    for (pos, &x) in view.iter().enumerate() {
        match x.try_into() {
            Ok(v) => out.push(v),
            Err(_) => {
                return Err(PyValueError::new_err(format!(
                    "{name}[{pos}] = {x} is negative or too large"
                )))
            }
        }
    }
    Ok(out)
}

fn convert_index_array<U>(arr: &IndexArray<'_>, name: &str) -> PyResult<Vec<U>>
where
    i32: TryInto<U>,
    i64: TryInto<U>,
{
    match arr {
        IndexArray::I32(a) => convert_indices(a, name),
        IndexArray::I64(a) => convert_indices(a, name),
    }
}

/// Fit SLIM item-item weights on a CSR user-item matrix.
///
/// `data`, `indices`, `indptr` are the CSR arrays of the (n_rows x n_cols)
/// user-item matrix; `data` must be float64, finite and non-negative;
/// `indices`/`indptr` may be int32 or int64. Returns `(indptr, indices, data)`
/// of the item-item weight matrix `W` in CSC layout (int64, int64, float64):
/// `indices[indptr[i]:indptr[i+1]]` are the neighbours `k` of target item `i`
/// (ascending) with weights `W[k, i] > 0`, so that `scores = X @ W`.
/// Raises `ValueError` on malformed input or out-of-range parameters.
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, n_rows, n_cols, lambd=0.5, beta=0.5, max_iter=100, tol=1e-6, n_threads=None))]
#[allow(clippy::too_many_arguments)]
fn solve_slim<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    indices: IndexArray<'py>,
    indptr: IndexArray<'py>,
    n_rows: usize,
    n_cols: usize,
    lambd: f64,
    beta: f64,
    max_iter: usize,
    tol: f64,
    n_threads: Option<usize>,
) -> PyResult<CscArrays<'py>> {
    // Borrow contiguous data in place; copy only strided views.
    let data: Cow<'_, [f64]> = match data.as_slice() {
        Ok(slice) => Cow::Borrowed(slice),
        Err(_) => Cow::Owned(data.as_array().to_vec()),
    };
    let indices: Vec<u32> = convert_index_array(&indices, "indices")?;
    let indptr: Vec<usize> = convert_index_array(&indptr, "indptr")?;
    let params = SlimParams {
        lambd,
        beta,
        max_iter,
        tol,
        n_threads,
    };
    let out = py
        .allow_threads(|| solve_slim_csr(&data, &indices, &indptr, n_rows, n_cols, params))
        .map_err(|e| match e {
            SlimError::InvalidInput(msg) => PyValueError::new_err(msg),
            SlimError::ThreadPool(_) => PyRuntimeError::new_err(e.to_string()),
        })?;
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
