// Dense helpers are written index-by-index on purpose to mirror the math.
#![allow(clippy::needless_range_loop)]

use super::*;
use crate::testing::{csr_from_dense, random_binary};

fn dense_gram(x: &[Vec<f64>], n_cols: usize) -> Vec<Vec<f64>> {
    let mut p = vec![vec![0.0; n_cols]; n_cols];
    for row in x {
        for a in 0..n_cols {
            for b in 0..n_cols {
                p[a][b] += row[a] * row[b];
            }
        }
    }
    p
}

fn to_dense(g: &Gram) -> Vec<Vec<f64>> {
    let mut p = vec![vec![0.0; g.n]; g.n];
    for k in 0..g.n {
        p[k][k] = g.diag[k];
        let (idx, val) = g.row(k);
        for (&j, &v) in idx.iter().zip(val) {
            p[k][j as usize] = v;
        }
    }
    p
}

fn assert_close(a: &[Vec<f64>], b: &[Vec<f64>]) {
    assert_eq!(a.len(), b.len());
    for (ra, rb) in a.iter().zip(b) {
        for (x, y) in ra.iter().zip(rb) {
            assert!((x - y).abs() < 1e-12, "{x} != {y}");
        }
    }
}

fn check_invariants(g: &Gram) {
    assert_eq!(g.indptr.len(), g.n + 1);
    assert_eq!(g.indptr[0], 0);
    assert_eq!(*g.indptr.last().unwrap(), g.indices.len());
    assert_eq!(g.indices.len(), g.data.len());
    assert_eq!(g.diag.len(), g.n);
    for k in 0..g.n {
        assert!(g.indptr[k] <= g.indptr[k + 1]);
        let (idx, val) = g.row(k);
        assert!(idx.windows(2).all(|w| w[0] < w[1]), "row {k} not sorted");
        assert!(idx.iter().all(|&j| j as usize != k), "row {k} has self");
        assert!(val.iter().all(|&v| v != 0.0), "row {k} stores a zero");
    }
}

#[test]
fn matches_dense_with_duplicates_and_empty_row_and_column() {
    // 4 users x 4 items: row 0 has a duplicate column 0 (1.0 + 0.5),
    // row 1 is empty, row 2 is unsorted, item 3 is never consumed.
    let data = vec![1.0, 2.0, 0.5, 1.0, 1.0, 3.0];
    let indices = vec![0u32, 1, 0, 2, 1, 0];
    let indptr = vec![0usize, 3, 3, 5, 6];
    let g = Gram::from_csr(&data, &indices, &indptr, 4, 4);
    let x = vec![
        vec![1.5, 2.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 1.0, 0.0],
        vec![3.0, 0.0, 0.0, 0.0],
    ];
    check_invariants(&g);
    assert_close(&to_dense(&g), &dense_gram(&x, 4));
    assert_eq!(g.diag, vec![11.25, 5.0, 1.0, 0.0]);
    assert_eq!(g.row(0).0, &[1]); // P_02 == 0 is not stored
    assert_eq!(g.row(3), (&[][..], &[][..]));
}

#[test]
fn matches_dense_random_binary() {
    let x = random_binary(7, 30, 12, 0.3);
    let (data, indices, indptr) = csr_from_dense(&x);
    let g = Gram::from_csr(&data, &indices, &indptr, 30, 12);
    check_invariants(&g);
    assert_close(&to_dense(&g), &dense_gram(&x, 12));
}

#[test]
fn symmetric_bitwise_without_duplicates() {
    let x = random_binary(11, 80, 20, 0.2);
    let (data, indices, indptr) = csr_from_dense(&x);
    let g = Gram::from_csr(&data, &indices, &indptr, 80, 20);
    let p = to_dense(&g);
    for a in 0..20 {
        for b in 0..20 {
            assert_eq!(p[a][b].to_bits(), p[b][a].to_bits());
        }
    }
}

#[test]
fn empty_inputs() {
    let g = Gram::from_csr(&[], &[], &[0], 0, 3);
    check_invariants(&g);
    assert_eq!(g.diag, vec![0.0; 3]);
    assert_eq!(g.indptr, vec![0; 4]);
    let g = Gram::from_csr(&[], &[], &[0, 0], 1, 0);
    assert_eq!(g.indptr, vec![0]);
    assert!(g.diag.is_empty());
}

#[test]
fn deterministic_across_thread_counts() {
    let x = random_binary(3, 200, 40, 0.1);
    let (data, indices, indptr) = csr_from_dense(&x);
    let build = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| Gram::from_csr(&data, &indices, &indptr, 200, 40))
    };
    assert_eq!(build(1), build(4));
}
