// Link directives for the LAPACK/BLAS backend.
//
// Features (select at most one):
//   accelerate  Apple Accelerate framework, new LAPACK interface
//               (`<name>$NEWLAPACK`, macOS 13.3+; default on macOS)
//   accelerate-legacy
//               Accelerate with the legacy `<name>_` symbols (older macOS)
//   openblas    OpenBLAS; library directory from $OPENBLAS_DIR/lib, else
//               /opt/homebrew/opt/openblas/lib on macOS, else the linker path
//   system      -llapack -lblas (default elsewhere)
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=OPENBLAS_DIR");
    println!("cargo:rustc-check-cfg=cfg(librla_newlapack)");

    let accelerate = env::var_os("CARGO_FEATURE_ACCELERATE").is_some();
    let legacy = env::var_os("CARGO_FEATURE_ACCELERATE_LEGACY").is_some();
    let openblas = env::var_os("CARGO_FEATURE_OPENBLAS").is_some();
    let system = env::var_os("CARGO_FEATURE_SYSTEM").is_some();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    let selected = [accelerate, openblas, system].iter().filter(|&&f| f).count();
    if selected > 1 {
        panic!("librla: select at most one backend feature (accelerate, openblas, system)");
    }

    let backend = if accelerate {
        "accelerate"
    } else if openblas {
        "openblas"
    } else if system {
        "system"
    } else if target_os == "macos" {
        "accelerate"
    } else {
        "system"
    };

    match backend {
        "accelerate" => {
            if target_os != "macos" {
                panic!("librla: the accelerate feature requires macOS");
            }
            println!("cargo:rustc-link-lib=framework=Accelerate");
            if !legacy {
                println!("cargo:rustc-cfg=librla_newlapack");
            }
        }
        "openblas" => {
            let dir = env::var("OPENBLAS_DIR").ok().or_else(|| {
                (target_os == "macos").then(|| "/opt/homebrew/opt/openblas".to_string())
            });
            if let Some(dir) = dir {
                let lib = Path::new(&dir).join("lib");
                println!("cargo:rustc-link-search=native={}", lib.display());
            }
            println!("cargo:rustc-link-lib=dylib=openblas");
        }
        _ => {
            println!("cargo:rustc-link-lib=dylib=lapack");
            println!("cargo:rustc-link-lib=dylib=blas");
        }
    }
}
