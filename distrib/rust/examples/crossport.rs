// Cross-port check: prints the CROSSPORT lines shared by the C++, Rust,
// Swift and Fortran ports (same LAPACK RNG and seed, so identical output up
// to LAPACK rounding). Run with `cargo run --example crossport`.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

use librla::{id_sketch, orth_sketch, svd_sketch, Complex64, Mat, Options, Rng};

fn ar(m: usize, n: usize) -> Mat<f64> {
    Mat::from_fn(m, n, |i, j| 1.0 / (i + j + 1) as f64)
}

fn ac(m: usize, n: usize) -> Mat<Complex64> {
    Mat::from_fn(m, n, |i, j| {
        Complex64::new(1.0, (i as f64 - j as f64) / 80.0) / (i + j + 1) as f64
    })
}

fn seed() -> Rng {
    Rng::new(1, 2, 3, 5).expect("valid seed")
}

/// C `%.15e` formatting (Rust's `{:.15e}` writes `1.5e0` instead of `1.5e+00`).
fn e15(v: f64) -> String {
    let s = format!("{v:.15e}");
    let (mant, exp) = s.split_once('e').expect("exponent");
    let exp: i32 = exp.parse().expect("integer exponent");
    let sign = if exp < 0 { '-' } else { '+' };
    format!("{mant}e{sign}{:02}", exp.abs())
}

fn join(v: impl IntoIterator<Item = String>) -> String {
    v.into_iter().collect::<Vec<_>>().join(" ")
}

/// The five CROSSPORT lines.
pub fn crossport_lines() -> Vec<String> {
    let mut lines = Vec::new();

    let mut rng = seed();
    let o = orth_sketch(&ar(50, 30), 5.0, Options::new().extra_samples(3).rng(&mut rng))
        .expect("orth_real");
    lines.push(format!("CROSSPORT orth_real {}", join(o.diag_r.iter().map(|&v| e15(v)))));
    let seed_after = rng.iseed();

    let mut rng = seed();
    let o = orth_sketch(&ac(50, 30), 5.0, Options::new().extra_samples(3).rng(&mut rng))
        .expect("orth_cplx");
    lines.push(format!("CROSSPORT orth_cplx {}", join(o.diag_r.iter().map(|&v| e15(v)))));

    lines.push(format!("CROSSPORT seed_after {}", join(seed_after.iter().map(|v| v.to_string()))));

    let mut rng = seed();
    let id = id_sketch(&ar(300, 200), 1e-10, Options::new().rng(&mut rng)).expect("id_real");
    lines.push(format!(
        "CROSSPORT id_real {} {}",
        id.k,
        join(id.piv[..10].iter().map(|v| v.to_string()))
    ));

    let mut rng = seed();
    let svd = svd_sketch(&ac(120, 80), 10.0, Options::new().power_iter(1).rng(&mut rng))
        .expect("svd_cplx");
    lines.push(format!(
        "CROSSPORT svd_cplx {} {}",
        svd.s.len(),
        join(svd.s[..5].iter().map(|&v| e15(v)))
    ));

    lines
}

fn main() {
    for line in crossport_lines() {
        println!("{line}");
    }
}
