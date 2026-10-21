// qr_sketch: both modes, tall and wide, deterministic fallback.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

#[macro_use]
mod common;

use common::{check, decaying, is_perm, kind, lower_max, ortho_err, rel, seed};
use librla::{dense, qr_sketch, Mat, Options, QrSketch, Scalar};

fn qr_err<T: Scalar>(a: &Mat<T>, r: &QrSketch<T>) -> f64 {
    rel(&a.select_columns(&r.p), &r.q.matmul(&r.r))
}

/// Error of the truncated deterministic pivoted QR.
fn qrpiv_err<T: Scalar>(a: &Mat<T>, k: usize) -> f64 {
    let (q, r, p) = dense::qr_pivoted(a).unwrap();
    rel(&a.select_columns(&p), &q.columns(0..k).matmul(&r.rows(0..k)))
}

fn shape_ok<T: Scalar>(a: &Mat<T>, r: &QrSketch<T>) -> bool {
    let (m, n) = a.shape();
    let k = r.q.ncols();
    r.q.nrows() == m
        && r.r.shape() == (k, n)
        && is_perm(&r.p, n)
        && lower_max(&r.r) == 0.0
        && ortho_err(&r.q) < 1e-13
}

fn modes<T: Scalar>() {
    let mut rng = seed();
    for (m, n) in [(400, 300), (150, 300)] {
        let a = decaying::<T>(&mut rng, m, n);
        let rtol = 1e-8;
        let r = qr_sketch(&a, rtol, Options::new().rng(&mut rng)).unwrap();
        let e = qr_err(&a, &r);
        check(
            &format!("qr tolerance {m}x{n} ({})", kind::<T>()),
            shape_ok(&a, &r) && e < 100.0 * rtol,
            format!("k={} err={e:.2e}", r.q.ncols()),
        );
        let k = 20;
        let r = qr_sketch(&a, k as f64, Options::new().rng(&mut rng)).unwrap();
        let (e, eref) = (qr_err(&a, &r), qrpiv_err(&a, k));
        check(
            &format!("qr rank {k} {m}x{n} ({})", kind::<T>()),
            shape_ok(&a, &r) && r.q.ncols() == k && e < 4.0 * eref,
            format!("err={e:.2e} qrpiv={eref:.2e}"),
        );
    }
}
both!(qr_modes, modes);

fn fallback<T: Scalar>() {
    let mut rng = seed();
    let a = rng.uniform::<T>(60, 50);
    let rtol = 1e-3;
    let r = qr_sketch(&a, rtol, Options::new().rng(&mut rng)).unwrap();
    let (q, rr, p) = dense::qr_pivoted(&a).unwrap();
    let d0 = rr[(0, 0)].abs();
    let k = (0..50).filter(|&i| rr[(i, i)].abs() >= rtol * d0).count();
    check(
        &format!("qr fallback deterministic ({})", kind::<T>()),
        r.p == p && r.q == q.columns(0..k) && r.r == rr.rows(0..k) && shape_ok(&a, &r),
        format!("k={k} err={:.2e}", qr_err(&a, &r)),
    );
}
both!(qr_fallback, fallback);
