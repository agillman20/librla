// RNG reference values (shared by all ports), seeding and reproducibility.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

#[macro_use]
mod common;

use common::{check, kind, seed};
use librla::{svd_sketch, Complex64, Error, Mat, Options, Rng, Scalar};

#[allow(clippy::excessive_precision)] // digits as given in the shared port spec
const REF: [f64; 6] = [
    3.7327920546847082E-01,
    8.2093410748050388E-01,
    5.5866811353917711E-01,
    6.4291221902741569E-01,
    6.8760847451716955E-01,
    1.6449965895444763E-01,
];
const REF_SEED_AFTER: [i32; 4] = [2384, 3667, 635, 1229];

#[test]
fn dlarnv_reference() {
    let mut rng = seed();
    let mut x = [0.0f64; 6];
    rng.fill(&mut x);
    let err = x.iter().zip(REF).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    check("rng dlarnv reference values", err <= 1e-15, format!("max diff {err:.1e}"));
    check("rng dlarnv seed after", rng.iseed() == REF_SEED_AFTER, format!("{:?}", rng.iseed()));
}

#[test]
fn zlarnv_reference() {
    let mut rng = seed();
    let mut z = [Complex64::new(0.0, 0.0); 3];
    rng.fill(&mut z);
    let flat: Vec<f64> = z.iter().flat_map(|c| [c.re, c.im]).collect();
    let err = flat.iter().zip(REF).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    check("rng zlarnv reference values", err <= 1e-15, format!("max diff {err:.1e}"));
    check("rng zlarnv seed after", rng.iseed() == REF_SEED_AFTER, format!("{:?}", rng.iseed()));
}

#[test]
fn constructors() {
    check("rng default seed", Rng::default().iseed() == [0, 0, 0, 1], "[0, 0, 0, 1]");
    let bad = [Rng::new(0, 0, 0, 2), Rng::new(-1, 0, 0, 1), Rng::new(4096, 0, 0, 1)];
    check(
        "rng invalid seeds rejected",
        bad.iter().all(|r| matches!(r, Err(Error::InvalidSeed(_)))),
        "even last / negative / > 4095",
    );
    let s: u64 = (1 << 36) | (2 << 24) | (3 << 12) | 4;
    check("rng from_seed mapping", Rng::from_seed(s).iseed() == [1, 2, 3, 5], format!("{s}"));
    check(
        "rng from_seed masks to 12 bits",
        Rng::from_seed(u64::MAX).iseed() == [4095; 4],
        "u64::MAX",
    );
}

fn reproducible<T: Scalar>() {
    let a = Mat::<T>::from_fn(80, 60, |i, j| T::from_real(1.0 / (i + j + 1) as f64));
    let run = |rng: Option<&mut Rng>| {
        let opts = Options { rng, ..Default::default() };
        svd_sketch(&a, 10.0, opts).unwrap()
    };
    let (mut r1, mut r2) = (seed(), seed());
    let (s1, s2) = (run(Some(&mut r1)), run(Some(&mut r2)));
    check(
        &format!("seed reproducible ({})", kind::<T>()),
        s1.u == s2.u && s1.s == s2.s && s1.vh == s2.vh,
        "same seed, identical U, s, Vh",
    );
    check(
        &format!("seed advances ({})", kind::<T>()),
        r1.iseed() != seed().iseed() && r1 == r2,
        format!("{:?}", r1.iseed()),
    );
    let s3 = run(Some(&mut r1));
    check(&format!("advanced rng differs ({})", kind::<T>()), s3.u != s1.u, "second call");
    let (d1, d2) = (run(None), run(Some(&mut Rng::default())));
    check(
        &format!("default rng deterministic ({})", kind::<T>()),
        d1.u == d2.u && d1.s == d2.s,
        "None == Rng::default()",
    );
}
both!(reproducibility, reproducible);
