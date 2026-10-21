// id_sketch and id_qrpiv: all methods, both modes, rank-deficient, k = 0, k = n.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

#[macro_use]
mod common;

use common::{check, decaying, id_err, is_perm, kind, optimal_err, seed, with_svals};
use librla::{id_qrpiv, id_sketch, Error, IdDecomp, Mat, Method, Options, Rng, Scalar};

const METHODS: [Method; 3] = [Method::Fast, Method::Svd, Method::Lstsq];

fn run<T: Scalar>(
    a: &Mat<T>, rtol: f64, method: Method, sketch: bool, rng: &mut Rng,
) -> IdDecomp<T> {
    if sketch {
        id_sketch(a, rtol, Options::new().method(method).rng(rng)).unwrap()
    } else {
        id_qrpiv(a, rtol, method).unwrap()
    }
}

fn well_formed<T: Scalar>(a: &Mat<T>, r: &IdDecomp<T>) -> bool {
    let n = a.ncols();
    is_perm(&r.piv, n) && r.t.shape() == (r.k, n - r.k) && r.t.is_finite()
}

fn name(sketch: bool) -> &'static str {
    if sketch {
        "id_sketch"
    } else {
        "id_qrpiv"
    }
}

fn modes<T: Scalar>() {
    let mut rng = seed();
    let a = decaying::<T>(&mut rng, 300, 200);
    let rtol = 1e-8;
    let k = 20;
    let opt = optimal_err(&a, k);
    for sketch in [true, false] {
        for method in METHODS {
            let r = run(&a, rtol, method, sketch, &mut rng);
            let e = id_err(&a, &r);
            check(
                &format!("{} tolerance {method} ({})", name(sketch), kind::<T>()),
                well_formed(&a, &r) && e < 100.0 * rtol,
                format!("k={} err={e:.2e}", r.k),
            );
            let r = run(&a, k as f64, method, sketch, &mut rng);
            let e = id_err(&a, &r);
            check(
                &format!("{} rank {k} {method} ({})", name(sketch), kind::<T>()),
                well_formed(&a, &r) && r.k == k && e < 100.0 * opt,
                format!("err={e:.2e} opt={opt:.2e}"),
            );
        }
    }
}
both!(id_modes, modes);

fn rank_deficient<T: Scalar>() {
    // True rank 5, requested rank 10: R11 is singular, T must stay finite.
    let mut rng = seed();
    let a = with_svals::<T>(&mut rng, 60, 40, &[1.0, 0.5, 0.25, 0.125, 0.0625]);
    for sketch in [true, false] {
        for method in METHODS {
            let r = run(&a, 10.0, method, sketch, &mut rng);
            let e = id_err(&a, &r);
            check(
                &format!("{} rank-deficient 10 > 5 {method} ({})", name(sketch), kind::<T>()),
                well_formed(&a, &r) && r.k == 10 && e < 1e-12,
                format!("err={e:.2e} max|T|={:.2e}", r.t.max_abs()),
            );
        }
    }
}
both!(id_rank_deficient, rank_deficient);

fn edge_cases<T: Scalar>() {
    let mut rng = seed();
    let z = Mat::<T>::zeros(60, 50);
    for sketch in [true, false] {
        for method in METHODS {
            let r = run(&z, 1e-6, method, sketch, &mut rng);
            check(
                &format!("{} zero matrix k=0 {method} ({})", name(sketch), kind::<T>()),
                r.k == 0 && r.t.shape() == (0, 50) && is_perm(&r.piv, 50),
                format!("k={}", r.k),
            );
        }
    }
    let a = rng.uniform::<T>(30, 10);
    for sketch in [true, false] {
        for method in METHODS {
            let r = run(&a, 10.0, method, sketch, &mut rng);
            check(
                &format!("{} k == n {method} ({})", name(sketch), kind::<T>()),
                r.k == 10 && r.t.shape() == (10, 0) && is_perm(&r.piv, 10),
                format!("k={}", r.k),
            );
        }
    }
}
both!(id_edge, edge_cases);

#[test]
fn invalid_method() {
    let bad = ["foo", "FAST", ""]
        .iter()
        .all(|s| matches!(s.parse::<Method>(), Err(Error::InvalidMethod(_))));
    let good = ["fast", "svd", "lstsq"]
        .iter()
        .all(|s| s.parse::<Method>().map(|m| m.as_str() == *s).unwrap_or(false));
    check("id invalid method rejected", bad && good, "'foo', 'FAST', '' -> Error::InvalidMethod");
}
