// Cross-port check: prints the CROSSPORT lines (run with --nocapture) and
// compares them against the values this port produced with Accelerate.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

#[allow(dead_code)]
#[path = "../examples/crossport.rs"]
mod crossport;

mod common;

use common::check;

/// Reference lines (Accelerate, new LAPACK interface; orth_real is bitwise
/// equal to Python/SciPy). Integers must match exactly; floating-point values
/// to a relative 1e-9 (LAPACK backends differ in the last digits).
const REFERENCE: [&str; 5] = [
    "CROSSPORT orth_real 2.201415876390088e+00 7.782393081571191e-01 1.792111547918920e-01 2.735372994185338e-02 1.829391246767265e-03 4.964313308298363e-04 4.086054533178452e-05 4.297896799911683e-06",
    "CROSSPORT orth_cplx 2.764954393717040e+00 8.441207939408220e-01 1.423567755739453e-01 3.392840380629036e-02 4.897016475004854e-03 2.911400935583974e-04 6.229146987168865e-05 1.610490859122329e-05",
    "CROSSPORT seed_after 3844 2983 1707 1605",
    "CROSSPORT id_real 16 0 3 28 1 168 10 72 5 199 2",
    "CROSSPORT svd_cplx 10 2.256748472841064e+00 8.299098477167930e-01 1.834095666318948e-01 3.407572079164049e-02 1.991850185385876e-02",
];

fn matches(line: &str, reference: &str) -> bool {
    let (a, b): (Vec<&str>, Vec<&str>) =
        (line.split_whitespace().collect(), reference.split_whitespace().collect());
    a.len() == b.len()
        && a.iter().zip(&b).all(|(x, y)| {
            if y.contains('.') {
                match (x.parse::<f64>(), y.parse::<f64>()) {
                    (Ok(u), Ok(v)) => (u - v).abs() <= 1e-9 * v.abs(),
                    _ => false,
                }
            } else {
                x == y
            }
        })
}

#[test]
fn crossport_lines() {
    let lines = crossport::crossport_lines();
    for line in &lines {
        println!("{line}");
    }
    for (line, reference) in lines.iter().zip(REFERENCE) {
        let name = line.split_whitespace().nth(1).unwrap_or("?");
        check(&format!("crossport {name} matches reference"), matches(line, reference), "rel 1e-9");
    }
    check("crossport five lines", lines.len() == 5, lines.len());
}
