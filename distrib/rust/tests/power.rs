// Power iteration on a slowly decaying spectrum.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

#[macro_use]
mod common;

use common::{check, kind, optimal_err, ortho_err, seed, svd_err, with_svals, Hidden};
use librla::{qr_sketch, svd_sketch, Options, Scalar};

fn power<T: Scalar>() {
    let mut rng = seed();
    let s: Vec<f64> = (0..200).map(|j| 1.0 / (j as f64 + 1.0)).collect();
    let a = with_svals::<T>(&mut rng, 300, 200, &s);
    let k = 20;
    let opt = optimal_err(&a, k);
    let errs: Vec<f64> = (0..3)
        .map(|p| {
            let mut r = seed();
            let svd = svd_sketch(&a, k as f64, Options::new().power_iter(p).rng(&mut r)).unwrap();
            assert!(ortho_err(&svd.u) < 1e-13);
            svd_err(&a, &svd)
        })
        .collect();
    check(
        &format!("power_iter 0,1,2 improve or keep svd accuracy ({})", kind::<T>()),
        errs[1] <= errs[0] * 1.001 && errs[2] <= errs[1] * 1.001 && errs[2] < 1.05 * opt,
        format!("err={:.4e} {:.4e} {:.4e} opt={opt:.4e}", errs[0], errs[1], errs[2]),
    );

    // Same through the matrix-free path, in tolerance mode as well
    let mut r = seed();
    let q = qr_sketch(&Hidden(&a), 0.05, Options::new().power_iter(2).rng(&mut r)).unwrap();
    let e = common::rel(&a.select_columns(&q.p), &q.q.matmul(&q.r));
    check(
        &format!("power_iter 2 matrix-free qr tolerance ({})", kind::<T>()),
        e < 0.2 && ortho_err(&q.q) < 1e-13,
        format!("k={} err={e:.2e}", q.q.ncols()),
    );
}
both!(power_iteration, power);
