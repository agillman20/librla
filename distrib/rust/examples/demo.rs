// Basic librla demo: dense Hilbert matrix and a matrix-free operator.
// Run with `cargo run --release --example demo`.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

use librla::{
    id_qrpiv, id_sketch, qr_sketch, svd_sketch, Complex64, FnOperator, Mat, Method, Options, Rng,
};

fn main() -> Result<(), librla::Error> {
    // Dense: 1000 x 500 Hilbert matrix (rapidly decaying singular values)
    let a = Mat::<f64>::from_fn(1000, 500, |i, j| 1.0 / (i + j + 1) as f64);
    let na = a.norm_fro();

    let svd = svd_sketch(&a, 1e-6, Options::default())?;
    let rec = svd.u.scale_columns(&svd.s).matmul(&svd.vh);
    println!(
        "svd_sketch  rtol=1e-6   k={:3}  err={:.2e}",
        svd.s.len(),
        (&a - &rec).norm_fro() / na
    );

    let qr = qr_sketch(&a, 20.0, Options::new().power_iter(1))?;
    let err = (&a.select_columns(&qr.p) - &qr.q.matmul(&qr.r)).norm_fro() / na;
    println!("qr_sketch   rank=20     k={:3}  err={err:.2e}", qr.q.ncols());

    let mut rng = Rng::from_seed(2026);
    let id = id_sketch(&a, 1e-6, Options::new().method(Method::Lstsq).rng(&mut rng))?;
    let skel = a.select_columns(&id.piv[..id.k]);
    let err = (&a.select_columns(&id.piv[id.k..]) - &skel.matmul(&id.t)).norm_fro() / na;
    println!("id_sketch   rtol=1e-6   k={:3}  err={err:.2e}  (lstsq)", id.k);

    let id = id_qrpiv(&a, 1e-6, Method::Fast)?;
    let skel = a.select_columns(&id.piv[..id.k]);
    let err = (&a.select_columns(&id.piv[id.k..]) - &skel.matmul(&id.t)).norm_fro() / na;
    println!("id_qrpiv    rtol=1e-6   k={:3}  err={err:.2e}", id.k);

    // Matrix-free: complex rank-10 product A = B C applied as B (C x)
    let (m, n, r) = (2000, 1500, 10);
    let b = rng.uniform::<Complex64>(m, r);
    let c = rng.uniform::<Complex64>(r, n);
    let op = FnOperator::new(
        m,
        n,
        |x: &Mat<Complex64>| b.matmul(&c.matmul(x)),
        |w: &Mat<Complex64>| c.adjoint_matmul(&b.adjoint_matmul(w)),
    );
    let svd = svd_sketch(&op, r as f64, Options::new().rng(&mut rng))?;
    let full = b.matmul(&c); // only to measure the error
    let rec = svd.u.scale_columns(&svd.s).matmul(&svd.vh);
    println!(
        "matrix-free complex {m}x{n} rank {r}: s[0]={:.4e} s[{}]={:.4e} err={:.2e}",
        svd.s[0],
        r - 1,
        svd.s[r - 1],
        (&full - &rec).norm_fro() / full.norm_fro()
    );
    Ok(())
}
