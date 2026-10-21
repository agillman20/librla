// orth_sketch: tolerance mode, rank mode, extra_samples, early termination.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

#[macro_use]
mod common;

use common::{check, decaying, kind, ortho_err, seed, span_err};
use librla::{orth_sketch, Error, Mat, Options, Scalar};

fn tolerance_mode<T: Scalar>() {
    let mut rng = seed();
    let a = decaying::<T>(&mut rng, 400, 300);
    let rtol = 1e-8;
    let r = orth_sketch(&a, rtol, Options::new().rng(&mut rng)).unwrap();
    let d = &r.diag_r;
    let accepted = d.len() > 12 && d[d.len() - 13] <= rtol * d[0];
    let sorted = d.windows(2).all(|w| w[1] <= w[0] * (1.0 + 1e-12));
    let (se, oe) = (span_err(&a, &r.q), ortho_err(&r.q));
    let name = format!("orth tolerance 400x300 ({})", kind::<T>());
    check(
        &name,
        r.flag == 0 && accepted && sorted && se < 100.0 * rtol && oe < 1e-13,
        format!("k={} span={se:.2e} ortho={oe:.1e}", r.q.ncols()),
    );
}
both!(orth_tolerance, tolerance_mode);

fn rank_mode<T: Scalar>() {
    let mut rng = seed();
    let a = decaying::<T>(&mut rng, 400, 300);
    let r = orth_sketch(&a, 20.0, Options::new().rng(&mut rng)).unwrap();
    let se = span_err(&a, &r.q.columns(0..20));
    check(
        &format!("orth rank 20 400x300 ({})", kind::<T>()),
        r.flag == 0
            && r.q.ncols() == 32
            && r.diag_r.len() == 32
            && ortho_err(&r.q) < 1e-13
            && se < 1e-4,
        format!("cols={} span(k=20)={se:.2e}", r.q.ncols()),
    );
    // Not capped by n: k + extra_samples > m gives min(m, k + extra) columns
    let b = decaying::<T>(&mut rng, 15, 40);
    let r = orth_sketch(&b, 10.0, Options::new().rng(&mut rng)).unwrap();
    check(
        &format!("orth rank 10 15x40 -> min(m, k+extra) ({})", kind::<T>()),
        r.flag == 0 && r.q.ncols() == 15,
        format!("cols={}", r.q.ncols()),
    );
    let r = orth_sketch(&a, 7.5, Options::new().extra_samples(3).rng(&mut rng)).unwrap();
    check(
        &format!("orth rank floor(7.5)+3 ({})", kind::<T>()),
        r.q.ncols() == 10,
        format!("cols={}", r.q.ncols()),
    );
}
both!(orth_rank, rank_mode);

fn legacy_extra0<T: Scalar>() {
    let mut rng = seed();
    let a = decaying::<T>(&mut rng, 400, 300);
    let rtol = 1e-8;
    let r = orth_sketch(&a, rtol, Options::new().extra_samples(0).rng(&mut rng)).unwrap();
    let d = &r.diag_r;
    check(
        &format!("orth extra_samples=0 legacy last-column test ({})", kind::<T>()),
        r.flag == 0 && r.q.ncols() == 42 && *d.last().unwrap() <= rtol * d[0],
        format!("cols={} last/first={:.2e}", r.q.ncols(), d.last().unwrap() / d[0]),
    );
}
both!(orth_legacy, legacy_extra0);

fn early_termination<T: Scalar>() {
    let mut rng = seed();
    let a = decaying::<T>(&mut rng, 400, 300);
    let r = orth_sketch(&a, 1e-17, Options::new().rng(&mut rng)).unwrap();
    check(
        &format!("orth flag 1 when rtol < eps ({})", kind::<T>()),
        r.flag == 1 && r.q.shape() == (400, 0) && r.diag_r.is_empty(),
        "empty Q",
    );
    let small = rng.uniform::<T>(30, 20);
    let r = orth_sketch(&small, 1e-8, Options::default()).unwrap();
    check(
        &format!("orth flag 1 when block_size >= min(m,n) ({})", kind::<T>()),
        r.flag == 1 && r.q.shape() == (30, 0),
        "30x20, block 42",
    );
    let full = rng.uniform::<T>(400, 300);
    let r = orth_sketch(&full, 1e-8, Options::new().rng(&mut rng)).unwrap();
    check(
        &format!("orth flag 1 when growth reaches min(m,n) ({})", kind::<T>()),
        r.flag == 1 && r.q.shape() == (400, 0) && r.diag_r.is_empty(),
        "full-rank 400x300",
    );
    let e = orth_sketch(&full, f64::NAN, Options::default());
    check(
        &format!("orth non-finite rtol is an error ({})", kind::<T>()),
        matches!(e, Err(Error::InvalidArgument(_))),
        "NaN",
    );
    let z = Mat::<T>::zeros(60, 50);
    let r = orth_sketch(&z, 1e-8, Options::default()).unwrap();
    check(
        &format!("orth zero matrix accepted ({})", kind::<T>()),
        r.flag == 0 && r.diag_r.iter().all(|&v| v == 0.0),
        format!("cols={}", r.q.ncols()),
    );
}
both!(orth_early, early_termination);
