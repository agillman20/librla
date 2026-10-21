// Matrix-free operators: a hidden dense wrapper matches the dense path (same
// seed), and a genuinely matrix-free low-rank product works in rank mode.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

#[macro_use]
mod common;

use common::{
    check, decaying, id_err, kind, ortho_err, rel, seed, span_err, svd_err, vec_diff, Hidden,
    LowRank,
};
use librla::{
    id_qrpiv, id_sketch, orth_sketch, qr_sketch, svd_sketch, Error, FnOperator, Mat, Method,
    Options, Scalar,
};

fn opts(rng: &mut librla::Rng) -> Options<'_> {
    Options::new().rng(rng)
}

fn qr_err<T: Scalar>(a: &Mat<T>, r: &librla::QrSketch<T>) -> f64 {
    rel(&a.select_columns(&r.p), &r.q.matmul(&r.r))
}

/// Errors of the dense and operator paths agree within a factor 2. The
/// sketch is identical, but `Q^H A` (dense) and `(A^H Q)^H` (operator) round
/// differently, which can flip near-tied pivots after the projection.
fn close(dense: f64, op: f64) -> bool {
    let slack = 1e-14;
    op <= 2.0 * dense + slack && dense <= 2.0 * op + slack
}

/// Run every routine on `a` dense and through `Hidden(a)` with the same seed.
/// orth_sketch, the materializing fallbacks and id_qrpiv must match exactly.
fn compare<T: Scalar>(a: &Mat<T>, rtol: f64, label: &str) {
    let h = Hidden(a);
    let tag = format!("{label} ({})", kind::<T>());

    let (mut r1, mut r2) = (seed(), seed());
    let d = orth_sketch(a, rtol, opts(&mut r1)).unwrap();
    let o = orth_sketch(&h, rtol, opts(&mut r2)).unwrap();
    check(
        &format!("linop orth matches dense {tag}"),
        d.flag == o.flag && d.q == o.q && d.diag_r == o.diag_r,
        format!("cols={} flag={}", o.q.ncols(), o.flag),
    );

    let (mut r1, mut r2) = (seed(), seed());
    let d = qr_sketch(a, rtol, opts(&mut r1)).unwrap();
    let o = qr_sketch(&h, rtol, opts(&mut r2)).unwrap();
    let (ed, eo) = (qr_err(a, &d), qr_err(a, &o));
    check(
        &format!("linop qr matches dense {tag}"),
        d.q.shape() == o.q.shape() && close(ed, eo) && ortho_err(&o.q) < 1e-13,
        format!("k={} err={eo:.2e} dense={ed:.2e} same_p={}", o.q.ncols(), d.p == o.p),
    );

    let (mut r1, mut r2) = (seed(), seed());
    let d = svd_sketch(a, rtol, opts(&mut r1)).unwrap();
    let o = svd_sketch(&h, rtol, opts(&mut r2)).unwrap();
    let ds = vec_diff(&d.s, &o.s);
    let de = (svd_err(a, &d) - svd_err(a, &o)).abs();
    check(
        &format!("linop svd matches dense {tag}"),
        ds < 1e-12 && de < 1e-10 && ortho_err(&o.u) < 1e-13,
        format!("k={} sdiff={ds:.1e} errdiff={de:.1e}", o.s.len()),
    );

    for method in [Method::Fast, Method::Svd, Method::Lstsq] {
        let (mut r1, mut r2) = (seed(), seed());
        let d = id_sketch(a, rtol, opts(&mut r1).method(method)).unwrap();
        let o = id_sketch(&h, rtol, opts(&mut r2).method(method)).unwrap();
        let (ed, eo) = (id_err(a, &d), id_err(a, &o));
        check(
            &format!("linop id_sketch {method} matches dense {tag}"),
            d.k == o.k && close(ed, eo) && o.t.is_finite(),
            format!("k={} err={eo:.2e} dense={ed:.2e} same_piv={}", o.k, d.piv == o.piv),
        );
        let d = id_qrpiv(a, rtol, method).unwrap();
        let o = id_qrpiv(&h, rtol, method).unwrap();
        check(
            &format!("linop id_qrpiv {method} matches dense {tag}"),
            d.k == o.k && d.piv == o.piv && d.t == o.t,
            format!("k={}", o.k),
        );
    }
}

fn hidden_dense<T: Scalar>() {
    let mut rng = seed();
    let a = decaying::<T>(&mut rng, 300, 200);
    compare(&a, 15.0, "rank 15 300x200");
    // 1.3e-8 lies between singular values (10^(-j/4)), so rounding cannot
    // decide the rank; 1e-8 would be exactly s_33
    compare(&a, 1.3e-8, "tolerance 300x200");
    // Full rank: tolerance mode terminates early and materializes the operator
    let b = rng.uniform::<T>(70, 60);
    compare(&b, 1e-6, "fallback 70x60");
    let w = rng.uniform::<T>(50, 70);
    compare(&w, 1e-6, "fallback wide 50x70");
    let w = decaying::<T>(&mut rng, 150, 300);
    compare(&w, 1.3e-8, "tolerance wide 150x300");
}
both!(linop_hidden_dense, hidden_dense);

fn matrix_free<T: Scalar>() {
    let mut rng = seed();
    let (m, n, r) = (500, 400, 15);
    let op = LowRank { b: rng.uniform::<T>(m, r), c: rng.uniform::<T>(r, n) };
    let a = op.b.matmul(&op.c); // for checking only
    let tag = format!("low-rank B C {m}x{n} r={r} ({})", kind::<T>());

    let o = orth_sketch(&op, r as f64, opts(&mut rng)).unwrap();
    let se = span_err(&a, &o.q);
    check(&format!("matfree orth {tag}"), se < 1e-13, format!("span={se:.1e}"));

    let q = qr_sketch(&op, r as f64, opts(&mut rng)).unwrap();
    let e = rel(&a.select_columns(&q.p), &q.q.matmul(&q.r));
    check(&format!("matfree qr {tag}"), q.q.ncols() == r && e < 1e-13, format!("err={e:.1e}"));

    let s = svd_sketch(&op, r as f64, opts(&mut rng)).unwrap();
    let e = svd_err(&a, &s);
    check(&format!("matfree svd {tag}"), s.s.len() == r && e < 1e-13, format!("err={e:.1e}"));

    // Wide matrix-free operator (adjoint path)
    let wide = LowRank { b: op.c.adjoint(), c: op.b.adjoint() };
    let s = svd_sketch(&wide, r as f64, opts(&mut rng)).unwrap();
    let e = svd_err(&a.adjoint(), &s);
    check(&format!("matfree svd wide {tag}"), s.s.len() == r && e < 1e-13, format!("err={e:.1e}"));

    for method in [Method::Fast, Method::Svd, Method::Lstsq] {
        let d = id_sketch(&op, r as f64, opts(&mut rng).method(method)).unwrap();
        let e = id_err(&a, &d);
        check(
            &format!("matfree id_sketch {method} {tag}"),
            d.k == r && e < 1e-12,
            format!("err={e:.1e}"),
        );
    }
}
both!(linop_matrix_free, matrix_free);

fn closures_and_errors<T: Scalar>() {
    let mut rng = seed();
    let a = decaying::<T>(&mut rng, 120, 90);
    let op = FnOperator::new(120, 90, |x: &Mat<T>| a.matmul(x), |w: &Mat<T>| a.adjoint_matmul(w));
    let s = svd_sketch(&op, 1e-6, Options::default()).unwrap();
    let e = svd_err(&a, &s);
    check(
        &format!("FnOperator svd tolerance ({})", kind::<T>()),
        e < 1e-4,
        format!("k={} err={e:.1e}", s.s.len()),
    );

    let bad = FnOperator::new(
        120,
        90,
        |x: &Mat<T>| Mat::zeros(119, x.ncols()),
        |w: &Mat<T>| a.adjoint_matmul(w),
    );
    let r = svd_sketch(&bad, 5.0, Options::default());
    check(
        &format!("operator shape mismatch is an error ({})", kind::<T>()),
        matches!(r, Err(Error::ShapeMismatch { call: "apply", .. })),
        r.err().map(|e| e.to_string()).unwrap_or_default(),
    );
}
both!(linop_closures, closures_and_errors);
