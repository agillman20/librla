// svd_sketch: both modes, tall and wide, singular values, fallback.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

#[macro_use]
mod common;

use common::{
    check, decaying, kind, optimal_err, ortho_err, ortho_err_rows, seed, svd_err, vec_diff,
};
use librla::{dense, svd_sketch, Mat, Options, Scalar, SvdSketch};

fn ortho_ok<T: Scalar>(r: &SvdSketch<T>) -> bool {
    ortho_err(&r.u) < 1e-13 && ortho_err_rows(&r.vh) < 1e-13
}

fn modes<T: Scalar>() {
    let mut rng = seed();
    for (m, n) in [(400, 300), (150, 300)] {
        let a = decaying::<T>(&mut rng, m, n);
        let s_true = dense::svd(&a).unwrap().1;

        let rtol = 1e-8;
        let r = svd_sketch(&a, rtol, Options::new().rng(&mut rng)).unwrap();
        let k_true = s_true.iter().filter(|&&v| v >= rtol * s_true[0]).count();
        let (e, ds) = (svd_err(&a, &r), vec_diff(&s_true[..r.s.len()], &r.s));
        check(
            &format!("svd tolerance {m}x{n} ({})", kind::<T>()),
            ortho_ok(&r)
                && e < 100.0 * rtol
                && ds < 1e-12
                && r.s.len().abs_diff(k_true) <= 1
                && r.u.shape() == (m, r.s.len())
                && r.vh.shape() == (r.s.len(), n),
            format!("k={} (true {k_true}) err={e:.2e} sdiff={ds:.1e}", r.s.len()),
        );

        let k = 20;
        let r = svd_sketch(&a, k as f64, Options::new().rng(&mut rng)).unwrap();
        let (e, opt) = (svd_err(&a, &r), optimal_err(&a, k));
        let ds = vec_diff(&s_true[..k], &r.s);
        check(
            &format!("svd rank {k} {m}x{n} ({})", kind::<T>()),
            ortho_ok(&r) && r.s.len() == k && e < 4.0 * opt && ds < 1e-6,
            format!("err={e:.2e} opt={opt:.2e} sdiff={ds:.1e}"),
        );
    }
}
both!(svd_modes, modes);

fn fallback<T: Scalar>() {
    let mut rng = seed();
    for (m, n) in [(60, 50), (50, 60)] {
        let a = rng.uniform::<T>(m, n);
        let rtol = 1e-3;
        let r = svd_sketch(&a, rtol, Options::new().rng(&mut rng)).unwrap();
        // Wide input goes through the adjoint, so compare against SVD(A^H).
        let (u, s, vh) = if m >= n {
            dense::svd(&a).unwrap()
        } else {
            let (u, s, vh) = dense::svd(&a.adjoint()).unwrap();
            (vh.adjoint(), s, u.adjoint())
        };
        let k = s.iter().filter(|&&v| v >= rtol * s[0]).count();
        let same = r.s == s[..k] && r.u == u.columns(0..k) && r.vh == vh.rows(0..k);
        check(
            &format!("svd fallback deterministic {m}x{n} ({})", kind::<T>()),
            same && ortho_ok(&r) && svd_err(&a, &r) < 1e-13,
            format!("k={k}"),
        );
    }
    let z = Mat::<T>::zeros(70, 60);
    let r = svd_sketch(&z, 1e-6, Options::default()).unwrap();
    check(&format!("svd zero matrix -> k=0 ({})", kind::<T>()), r.s.is_empty(), "70x60");
}
both!(svd_fallback, fallback);
