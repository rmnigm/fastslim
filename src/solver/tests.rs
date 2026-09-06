// Dense helpers are written index-by-index on purpose, to mirror the maths.
#![allow(clippy::needless_range_loop)]

use super::*;
use crate::testing::{csr_from_dense, near_duplicate_items, random_binary, random_zipf_counts};

fn dense_gram(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = x[0].len();
    let mut p = vec![vec![0.0; n]; n];
    for row in x {
        for a in 0..n {
            for b in 0..n {
                p[a][b] += row[a] * row[b];
            }
        }
    }
    p
}

/// Dense `W[k][i]` from CSC output.
fn to_dense_w(out: &CscOutput) -> Vec<Vec<f64>> {
    let n = out.n_items;
    let mut w = vec![vec![0.0; n]; n];
    for i in 0..n {
        for t in out.indptr[i] as usize..out.indptr[i + 1] as usize {
            w[out.indices[t] as usize][i] = out.data[t];
        }
    }
    w
}

/// Plain full-pass non-negative coordinate descent; returns dense `W[k][i]`.
fn reference(x: &[Vec<f64>], lambd: f64, beta: f64) -> Vec<Vec<f64>> {
    let p = dense_gram(x);
    let n = p.len();
    let mut wmat = vec![vec![0.0; n]; n];
    for i in 0..n {
        let mut w = vec![0.0; n];
        for _ in 0..100_000 {
            let mut max_delta = 0.0f64;
            for k in 0..n {
                if k == i {
                    continue;
                }
                let denom = p[k][k] + beta;
                if denom == 0.0 {
                    continue;
                }
                let mut c = p[i][k];
                for j in 0..n {
                    if j != k {
                        c -= p[k][j] * w[j];
                    }
                }
                let w_new = ((c - lambd) / denom).max(0.0);
                let d = w_new - w[k];
                if d != 0.0 {
                    w[k] = w_new;
                    max_delta = max_delta.max(d.abs());
                }
            }
            if max_delta < 1e-14 {
                break;
            }
        }
        for k in 0..n {
            wmat[k][i] = w[k];
        }
    }
    wmat
}

/// KKT violation of every coordinate of `W`, computed from the dense Gram.
fn kkt_violations(x: &[Vec<f64>], w: &[Vec<f64>], lambd: f64, beta: f64) -> Vec<Vec<f64>> {
    let p = dense_gram(x);
    let n = p.len();
    let mut viol = vec![vec![0.0; n]; n];
    for i in 0..n {
        for k in 0..n {
            if k == i {
                continue;
            }
            let mut g = p[i][k];
            for j in 0..n {
                g -= w[j][i] * p[j][k];
            }
            viol[i][k] = if w[k][i] > 0.0 {
                (g - lambd - beta * w[k][i]).abs()
            } else {
                (g - lambd).max(0.0)
            };
        }
    }
    viol
}

/// Largest KKT violation of `W`.
fn kkt_violation(x: &[Vec<f64>], w: &[Vec<f64>], lambd: f64, beta: f64) -> f64 {
    kkt_violations(x, w, lambd, beta)
        .iter()
        .flatten()
        .fold(0.0, |a, &b| f64::max(a, b))
}

fn fit(x: &[Vec<f64>], params: SlimParams) -> CscOutput {
    let (data, indices, indptr) = csr_from_dense(x);
    solve_slim_csr(&data, &indices, &indptr, x.len(), x[0].len(), params).unwrap()
}

fn tight(lambd: f64, beta: f64) -> SlimParams {
    SlimParams {
        lambd,
        beta,
        max_iter: 10_000,
        tol: 1e-10,
        n_threads: Some(1),
    }
}

fn check_invariants(out: &CscOutput) {
    let n = out.n_items;
    assert_eq!(out.n_passes.len(), n);
    assert_eq!(out.converged.len(), n);
    assert!(out.n_passes.iter().all(|&p| p >= 0));
    assert_eq!(out.indptr.len(), n + 1);
    assert_eq!(out.indptr[0], 0);
    assert_eq!(*out.indptr.last().unwrap() as usize, out.indices.len());
    assert_eq!(out.indices.len(), out.data.len());
    assert!(out.indptr.windows(2).all(|p| p[0] <= p[1]));
    for i in 0..n {
        let seg = out.indptr[i] as usize..out.indptr[i + 1] as usize;
        let idx = &out.indices[seg.clone()];
        assert!(idx.windows(2).all(|p| p[0] < p[1]), "segment {i} unsorted");
        assert!(idx.iter().all(|&k| k as usize != i), "self weight in {i}");
        assert!(
            out.data[seg].iter().all(|&w| w > 0.0),
            "non-positive weight"
        );
    }
}

fn max_abs_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f64::max)
}

fn nnz(w: &[Vec<f64>]) -> usize {
    w.iter().flatten().filter(|&&v| v > 0.0).count()
}

fn assert_matches_reference(x: &[Vec<f64>], lambd: f64, beta: f64) {
    let out = fit(x, tight(lambd, beta));
    check_invariants(&out);
    assert!(out.converged.iter().all(|&c| c), "not all items converged");
    let w = to_dense_w(&out);
    let w_ref = reference(x, lambd, beta);
    assert_eq!(
        nnz(&w),
        nnz(&w_ref),
        "nnz differs (lambd={lambd}, beta={beta})"
    );
    let diff = max_abs_diff(&w, &w_ref);
    assert!(
        diff < 1e-6,
        "max |W - W_ref| = {diff} (lambd={lambd}, beta={beta})"
    );
    let kkt = kkt_violation(x, &w, lambd, beta);
    assert!(
        kkt < 1e-6,
        "KKT violation {kkt} (lambd={lambd}, beta={beta})"
    );
}

fn three_items() -> Vec<Vec<f64>> {
    vec![
        vec![1.0, 1.0, 0.0],
        vec![1.0, 1.0, 1.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 0.0],
    ]
}

#[test]
fn hand_computed_three_items() {
    // P = [[3,2,1],[2,3,2],[1,2,2]], lambd = beta = 0.5.
    // item 0: w_1 = 1.5/3.5 = 3/7, w_2 clipped to 0 (KKT: 0.5 - 2*3/7 < 0)
    // item 1: [3.5 1; 1 2.5] w = [1.5; 1.5]  ->  w = [9/31, 15/31]
    // item 2: w_0 clipped to 0, w_1 = 1.5/3.5 = 3/7
    let out = fit(&three_items(), tight(0.5, 0.5));
    assert_eq!(out.indptr, vec![0, 1, 3, 4]);
    assert_eq!(out.indices, vec![1, 0, 2, 1]);
    let expected = [3.0 / 7.0, 9.0 / 31.0, 15.0 / 31.0, 3.0 / 7.0];
    for (got, want) in out.data.iter().zip(expected) {
        assert!((got - want).abs() < 1e-9, "{got} vs {want}");
    }
    assert_matches_reference(&three_items(), 0.5, 0.5);
}

#[test]
fn lambda_zero_matches_reference() {
    assert_matches_reference(&three_items(), 0.0, 0.5);
    assert_matches_reference(&random_binary(21, 40, 10, 0.25), 0.0, 1.0);
}

#[test]
fn beta_zero_matches_reference() {
    assert_matches_reference(&three_items(), 0.5, 0.0);
    assert_matches_reference(&random_binary(22, 40, 10, 0.25), 0.7, 0.0);
}

#[test]
fn huge_lambda_gives_empty() {
    let out = fit(&three_items(), tight(100.0, 0.5));
    check_invariants(&out);
    assert_eq!(out.indptr, vec![0, 0, 0, 0]);
    assert!(out.indices.is_empty() && out.data.is_empty());
    // No candidates: nothing to solve, trivially converged in 0 passes.
    assert_eq!(out.n_passes, vec![0, 0, 0]);
    assert_eq!(out.converged, vec![true, true, true]);
}

#[test]
fn max_iter_zero_gives_empty() {
    let mut p = tight(0.5, 0.5);
    p.max_iter = 0;
    let out = fit(&three_items(), p);
    check_invariants(&out);
    assert_eq!(out.indptr, vec![0, 0, 0, 0]);
    // Every item has candidates (all P_ik >= 0.5) but no budget.
    assert_eq!(out.n_passes, vec![0, 0, 0]);
    assert_eq!(out.converged, vec![false, false, false]);
}

#[test]
fn reports_pass_counts_and_convergence() {
    let x = random_zipf_counts(41, 300, 40);
    let params = |max_iter| SlimParams {
        lambd: 0.5,
        beta: 0.5,
        max_iter,
        tol: 1e-8,
        n_threads: Some(1),
    };

    // Tight budget: truncated items are flagged and report exactly max_iter.
    let cut = fit(&x, params(3));
    check_invariants(&cut);
    let n_bad = cut.converged.iter().filter(|&&c| !c).count();
    assert!(n_bad > 0, "expected some items to hit max_iter = 3");
    for i in 0..cut.n_items {
        if cut.converged[i] {
            assert!(cut.n_passes[i] <= 3, "item {i}: {} passes", cut.n_passes[i]);
        } else {
            assert_eq!(cut.n_passes[i], 3, "item {i}");
        }
    }

    // Generous budget: everything converges within it.
    let ok = fit(&x, params(10_000));
    check_invariants(&ok);
    assert!(ok.converged.iter().all(|&c| c));
    for i in 0..ok.n_items {
        assert!(ok.n_passes[i] <= 10_000);
        if ok.indptr[i + 1] > ok.indptr[i] {
            assert!(ok.n_passes[i] >= 1, "item {i}");
        }
    }
    assert!(
        ok.n_passes.iter().any(|&p| p > 3),
        "budget of 3 was not binding"
    );

    // tol = 0 disables the stopping test: exactly max_iter passes.
    let exact = fit(
        &x,
        SlimParams {
            tol: 0.0,
            ..params(7)
        },
    );
    assert!(exact.n_passes.iter().all(|&p| p == 7));
    assert!(exact.converged.iter().all(|&c| !c));
}

#[test]
fn converged_items_meet_kkt_bound() {
    // A converged exit bounds every KKT violation by (P_kk + beta) * tol; the
    // relative slack covers round-off in the maintained residuals.
    let problems = [
        random_binary(31, 60, 20, 0.3),
        random_zipf_counts(32, 400, 50),
        near_duplicate_items(34, 500, 12, 20),
        near_duplicate_items(35, 2000, 20, 10),
    ];
    for x in &problems {
        let p = dense_gram(x);
        for (lambd, beta) in [(0.5, 0.5), (0.1, 0.0), (2.0, 1.0)] {
            for tol in [1e-2, 1e-4, 1e-6] {
                let out = fit(
                    x,
                    SlimParams {
                        lambd,
                        beta,
                        max_iter: 1_000_000,
                        tol,
                        n_threads: Some(1),
                    },
                );
                check_invariants(&out);
                assert!(out.converged.iter().all(|&c| c), "tol={tol}");
                let viol = kkt_violations(x, &to_dense_w(&out), lambd, beta);
                for i in 0..out.n_items {
                    for k in 0..out.n_items {
                        let bound = (p[k][k] + beta) * tol;
                        assert!(
                            viol[i][k] <= bound * (1.0 + 1e-6),
                            "item {i}, coord {k}: violation {} > bound {bound} \
                             (lambd={lambd}, beta={beta}, tol={tol})",
                            viol[i][k]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn rejects_gram_overflow() {
    // 1e200^2 overflows P to +inf, which would turn every update into NaN.
    let (mut data, indices, indptr) = csr_from_dense(&three_items());
    for v in data.iter_mut() {
        *v = 1e200;
    }
    let err = solve_slim_csr(&data, &indices, &indptr, 4, 3, SlimParams::default()).unwrap_err();
    assert!(
        matches!(&err, SlimError::InvalidInput(msg) if msg.contains("overflow")),
        "{err}"
    );
    // Just below the threshold the same matrix solves fine.
    for v in data.iter_mut() {
        *v = 1e150;
    }
    let out = solve_slim_csr(&data, &indices, &indptr, 4, 3, SlimParams::default()).unwrap();
    assert!(out.data.iter().all(|v| v.is_finite()));
}

#[test]
fn zero_curvature_coordinate_is_left_at_zero() {
    // One user with x = [1e150, 1e-170]: P_00 = 1e300 and P_01 = 1e-20
    // are finite but P_11 = 1e-340 underflows to 0. With beta = 0 the
    // step for w_1 of item 0 would divide by zero; it must be skipped.
    let x = vec![vec![1e150, 1e-170]];
    let out = fit(
        &x,
        SlimParams {
            lambd: 0.0,
            beta: 0.0,
            max_iter: 100,
            tol: 1e-12,
            n_threads: Some(1),
        },
    );
    check_invariants(&out);
    assert!(out.data.iter().all(|v| v.is_finite()));
    assert_eq!(out.indptr[0], out.indptr[1], "item 0 must have no weights");
    assert!(out.converged.iter().all(|&c| c));
}

#[test]
fn item_with_small_norm_is_not_skipped() {
    // P_00 = 3 < lambd^2 = 4, yet P_01 = 3 >= lambd, so w_10 = 1/10.5 > 0.
    let mut x = vec![vec![0.0; 3]; 10];
    for (u, row) in x.iter_mut().enumerate() {
        if u < 3 {
            row[0] = 1.0;
        }
        row[1] = 1.0;
        if u >= 5 {
            row[2] = 1.0;
        }
    }
    let out = fit(&x, tight(2.0, 0.5));
    let w = to_dense_w(&out);
    assert!((w[1][0] - 1.0 / 10.5).abs() < 1e-9, "w_10 = {}", w[1][0]);
    assert_matches_reference(&x, 2.0, 0.5);
}

#[test]
fn kkt_on_random_binary_problems() {
    for seed in 1..=3u64 {
        let x = random_binary(seed, 40, 15, 0.2);
        for lambd in [0.1, 0.5, 1.5] {
            assert_matches_reference(&x, lambd, 0.5);
        }
    }
}

#[test]
fn deterministic_across_thread_counts() {
    let x = random_binary(5, 300, 60, 0.1);
    let run = |n_threads| {
        fit(
            &x,
            SlimParams {
                lambd: 0.5,
                beta: 0.5,
                max_iter: 100,
                tol: 1e-6,
                n_threads,
            },
        )
    };
    let one = run(Some(1));
    assert!(!one.data.is_empty());
    assert_eq!(one, run(Some(1)));
    assert_eq!(one, run(Some(4)));
    assert_eq!(one, run(None));
}

#[test]
fn output_invariants_on_random_problem() {
    let x = random_binary(9, 200, 50, 0.15);
    for lambd in [0.0, 0.5, 2.0] {
        let out = fit(&x, tight(lambd, 0.5));
        check_invariants(&out);
    }
}

#[test]
fn duplicates_are_summed_like_canonical_input() {
    // Replace the first stored entry of row 0 (value 1.0) by two entries
    // 0.75 + 0.25 for the same column; the result must not change.
    let x = random_binary(13, 20, 6, 0.4);
    let (data, indices, indptr) = csr_from_dense(&x);
    assert!(indptr[1] > 0, "test assumes row 0 is non-empty");
    let params = tight(0.3, 0.5);
    let canonical = solve_slim_csr(&data, &indices, &indptr, 20, 6, params).unwrap();
    let mut d2 = vec![0.75, 0.25];
    d2.extend_from_slice(&data[1..]);
    let mut i2 = vec![indices[0], indices[0]];
    i2.extend_from_slice(&indices[1..]);
    let p2: Vec<usize> = indptr
        .iter()
        .enumerate()
        .map(|(u, &p)| if u == 0 { 0 } else { p + 1 })
        .collect();
    let dup = solve_slim_csr(&d2, &i2, &p2, 20, 6, params).unwrap();
    assert_eq!(dup.indptr, canonical.indptr);
    assert_eq!(dup.indices, canonical.indices);
    assert!(max_abs_diff(&to_dense_w(&dup), &to_dense_w(&canonical)) < 1e-12);
}

#[test]
fn rejects_invalid_input() {
    let (data, indices, indptr) = csr_from_dense(&three_items());
    let ok = SlimParams::default();
    let bad = |d: &[f64], ix: &[u32], ip: &[usize], p: SlimParams| {
        matches!(
            solve_slim_csr(d, ix, ip, 4, 3, p),
            Err(SlimError::InvalidInput(_))
        )
    };
    assert!(solve_slim_csr(&data, &indices, &indptr, 4, 3, ok).is_ok());
    assert!(bad(&data, &indices, &indptr[..4], ok)); // indptr too short
    assert!(bad(&data, &indices, &[1, 2, 5, 7, 8], ok)); // indptr[0] != 0
    assert!(bad(&data, &indices, &[0, 5, 2, 7, 8], ok)); // non-monotone
    assert!(bad(&data, &indices, &[0, 2, 5, 7, 9], ok)); // indptr[n] != nnz
    assert!(bad(&data, &indices[..7], &indptr, ok)); // indices/data mismatch
    let mut oob = indices.to_vec();
    oob[0] = 3;
    assert!(bad(&data, &oob, &indptr, ok)); // index out of range
    let mut nan = data.to_vec();
    nan[1] = f64::NAN;
    assert!(bad(&nan, &indices, &indptr, ok));
    let mut neg = data.to_vec();
    neg[1] = -1.0;
    assert!(bad(&neg, &indices, &indptr, ok));
    for p in [
        SlimParams { lambd: -1.0, ..ok },
        SlimParams { beta: -1.0, ..ok },
        SlimParams { tol: -1.0, ..ok },
        SlimParams {
            lambd: f64::NAN,
            ..ok
        },
        SlimParams {
            n_threads: Some(0),
            ..ok
        },
    ] {
        assert!(bad(&data, &indices, &indptr, p), "{p:?} accepted");
    }
}
