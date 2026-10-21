// swift-tools-version: 6.0
//
// LibRLA - randomized low-rank factorizations (experimental Swift port)
//
// Backend selection (environment variables read when the manifest is evaluated):
//   (default)              macOS: Accelerate, LAPACK 3.11 `$NEWLAPACK` symbols
//                          (macOS 13.3+); Linux: -llapack -lblas
//   LIBRLA_BLAS=openblas   link OpenBLAS instead; on macOS it is looked up in
//                          LIBRLA_OPENBLAS_DIR (default /opt/homebrew/opt/openblas)
// Use a separate build directory per backend, e.g.
//   LIBRLA_BLAS=openblas swift test --build-path .build-openblas
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

import PackageDescription
import Foundation

let environment = ProcessInfo.processInfo.environment
let useOpenBLAS = environment["LIBRLA_BLAS"]?.lowercased() == "openblas"
let openBLASDir = environment["LIBRLA_OPENBLAS_DIR"] ?? "/opt/homebrew/opt/openblas"

let lapackLinkerSettings: [LinkerSetting] = useOpenBLAS
    ? [
        .linkedLibrary("openblas"),
        // OpenBLAS is keg-only in Homebrew; unsafe flags are only used in this opt-in mode.
        .unsafeFlags(["-L\(openBLASDir)/lib"], .when(platforms: [.macOS])),
    ]
    : [
        .linkedFramework("Accelerate", .when(platforms: [.macOS, .iOS, .tvOS, .watchOS, .visionOS])),
        .linkedLibrary("lapack", .when(platforms: [.linux])),
        .linkedLibrary("blas", .when(platforms: [.linux])),
    ]

let package = Package(
    name: "LibRLA",
    // macOS 14: Accelerate's $NEWLAPACK interface needs 13.3, swift-testing needs 14.
    // Homebrew's OpenBLAS dylib is built for the host macOS; match it to keep the link warning-free.
    platforms: [useOpenBLAS ? .macOS("26.0") : .macOS(.v14)],
    products: [
        .library(name: "LibRLA", targets: ["LibRLA"]),
        .executable(name: "librla-demo", targets: ["LibRLADemo"]),
        .executable(name: "librla-test", targets: ["LibRLATestRunner"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0"),
    ],
    targets: [
        .target(
            name: "CLapack",
            // Plain Fortran symbol names (no Accelerate $NEWLAPACK labels) for OpenBLAS and Linux.
            cSettings: useOpenBLAS ? [.define("LIBRLA_OPENBLAS")]
                                   : [.define("LIBRLA_OPENBLAS", .when(platforms: [.linux]))],
            linkerSettings: lapackLinkerSettings
        ),
        .target(
            name: "LibRLA",
            dependencies: [
                "CLapack",
                .product(name: "ComplexModule", package: "swift-numerics"),
                .product(name: "RealModule", package: "swift-numerics"),
            ]
        ),
        .target(name: "LibRLACheckSuite", dependencies: ["LibRLA", .product(name: "ComplexModule", package: "swift-numerics"), .product(name: "RealModule", package: "swift-numerics")], path: "test/Support"),
        .executableTarget(name: "LibRLATestRunner", dependencies: ["LibRLACheckSuite"], path: "test/Runner"),
        .testTarget(name: "LibRLATests", dependencies: ["LibRLACheckSuite", "LibRLA"], path: "test/Tests"),
        .executableTarget(name: "LibRLADemo", dependencies: ["LibRLA"], path: "demo"),
    ]
)
