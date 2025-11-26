#=
test_image.jl - Demonstrate image compression using truncated SVD.

Description
-----------
This script demonstrates low-rank image compression using the
randomized truncated SVD from librla. It loads a grayscale image,
computes a rank-k approximation, and displays the results.

Two compression methods are compared:
  1. Basic truncated SVD with rank k
  2. Truncated SVD with power iterations and extra samples for
     improved accuracy

Set use_single=true to run in single precision (faster, less memory).

Requirements
------------
* librla.jl in the path
* Image file: pexels-flickr-149387.jpg
* Packages: Images, FileIO, PyPlot

See also: librla.svd_sketch

Image source: https://www.pexels.com/photo/silver-metal-round-gears-connected-to-each-other-149387/
=#

using LinearAlgebra
using Images
using FileIO
using PyPlot

# Include librla from parent directory
include(joinpath(@__DIR__, "..", "julia", "librla.jl"))
using .librla

# Load image
A = load("pexels-flickr-149387.jpg")
# A = load("pexels-anniroenkae-4793404.jpg")
# A = load("x1d-II-sample-02.jpg")

# A = permutedims(A, (2, 1, 3))

# Convert to grayscale if RGB
if eltype(A) <: RGB
    A = Gray.(A)
end

# Convert to matrix of floats (0-255 range)
A_mat = Float64.(A) * 255

println("Image size: $(size(A_mat))")

figure(1)
imshow(A_mat, cmap="gray", aspect="equal")
colorbar()
title("Original (grayscale)")

k = 60 * 2
use_single = true  # set to true for single precision

if use_single
    conv = Float32
else
    conv = Float64
end

A_conv = conv.(A_mat)

t0 = time()
U, s, Vt = librla.svd_sketch(A_conv, k)
B = U * diagm(s) * Vt
println("Elapsed time: $(round(time() - t0, digits=3))s")

figure(2)
imshow(clamp.(B, 0, 255), cmap="gray", aspect="equal")
colorbar()
title("Rank-$k SVD approximation")

rel_error = norm(conv.(A_mat) - conv.(B)) / norm(conv.(A_mat))
println("Relative error: $(round(rel_error, sigdigits=6))")


t0 = time()
U, s, Vt = librla.svd_sketch(A_conv, k, power_iter=1, extra_samples=div(k, 4))
B = U * diagm(s) * Vt
println("Elapsed time: $(round(time() - t0, digits=3))s")

figure(3)
imshow(clamp.(B, 0, 255), cmap="gray", aspect="equal")
colorbar()
title("Rank-$k SVD (power iterations, extra sampling)")

rel_error = norm(conv.(A_mat) - conv.(B)) / norm(conv.(A_mat))
println("Relative error: $(round(rel_error, sigdigits=6))")

