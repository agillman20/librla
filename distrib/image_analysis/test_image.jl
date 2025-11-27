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
* Packages: Images, FileIO, GLMakie

See also: librla.svd_sketch

Image source: https://www.pexels.com/photo/silver-metal-round-gears-connected-to-each-other-149387/
=#

using LinearAlgebra
using Images
using FileIO
using GLMakie
using GLFW

# Include librla from parent directory
include(joinpath(@__DIR__, "..", "julia", "librla.jl"))
using .librla

# Load image
A = load("pexels-flickr-149387.jpg")
# A = load("pexels-anniroenkae-4793404.jpg")

# A = permutedims(A, (2, 1, 3))

# Convert to grayscale if RGB
if eltype(A) <: RGB
    A = Gray.(A)
end

# Convert to matrix of floats (0-255 range)
A_mat = Float64.(A) * 255

println("Image size: $(size(A_mat))")

fig1 = Figure(size=(800, 600))
ax1 = GLMakie.Axis(fig1[1,1], title="Original (grayscale)", aspect=DataAspect(), yreversed=true)
hm1 = heatmap!(ax1, A_mat', colormap=:grays)
Colorbar(fig1[1,2], hm1, minorticks=IntervalsBetween(5))
rowsize!(fig1.layout, 1, ax1.scene.viewport[].widths[2])
scr1 = display(GLMakie.Screen(title="Figure 1"), fig1)
GLFW.SetWindowPos(scr1.glscreen, 50, 50)

k = 60*2
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

fig2 = Figure(size=(800, 600))
ax2 = GLMakie.Axis(fig2[1,1], title="Rank-$k SVD approximation", aspect=DataAspect(), yreversed=true)
hm2 = heatmap!(ax2, clamp.(B, 0, 255)', colormap=:grays)
Colorbar(fig2[1,2], hm2, minorticks=IntervalsBetween(5))
rowsize!(fig2.layout, 1, ax2.scene.viewport[].widths[2])
scr2 = display(GLMakie.Screen(title="Figure 2"), fig2)
GLFW.SetWindowPos(scr2.glscreen, 100, 100)

rel_error = norm(conv.(A_mat) - conv.(B)) / norm(conv.(A_mat))
println("Relative error: $(round(rel_error, sigdigits=6))")


t0 = time()
U, s, Vt = librla.svd_sketch(A_conv, k, power_iter=1, extra_samples=div(k, 4))
B = U * diagm(s) * Vt
println("Elapsed time: $(round(time() - t0, digits=3))s")

fig3 = Figure(size=(800, 600))
ax3 = GLMakie.Axis(fig3[1,1], title="Rank-$k SVD (power iterations, extra sampling)", aspect=DataAspect(), yreversed=true)
hm3 = heatmap!(ax3, clamp.(B, 0, 255)', colormap=:grays)
Colorbar(fig3[1,2], hm3, minorticks=IntervalsBetween(5))
rowsize!(fig3.layout, 1, ax3.scene.viewport[].widths[2])
scr3 = display(GLMakie.Screen(title="Figure 3"), fig3)
GLFW.SetWindowPos(scr3.glscreen, 150, 150)

rel_error = norm(conv.(A_mat) - conv.(B)) / norm(conv.(A_mat))
println("Relative error: $(round(rel_error, sigdigits=6))")

