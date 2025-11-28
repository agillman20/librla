#=
test_image_rgb.jl - Demonstrate RGB image compression using truncated SVD.

Description
-----------
This script demonstrates low-rank image compression using the
randomized truncated SVD from librla. It loads an RGB image,
reshapes it to a 2D matrix (m x 3n) for SVD, and displays results.

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

See also: librla.svd_sketch, test_image.jl

Image source: https://www.pexels.com/photo/silver-metal-round-gears-connected-to-each-other-149387/
https://1.img-dpreview.com/files/p/sample_galleries/6044553814/1399808671.jpg
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
# A = load("pexels-flickr-149387.jpg")
# A = load("pexels-anniroenkae-4793404.jpg")
# A = load("b_29667.jpg")
A = load("pexels-andre-ulysses-de-salis-2100065-7824822.jpg")

# A = permutedims(A, (2, 1))

# Get dimensions
m, n = size(A)
nc = 3  # RGB channels

println("Image size: $m x $n x $nc")

fig1 = Figure()
ax1 = GLMakie.Axis(fig1[1,1], title="Original (RGB)", aspect=DataAspect(), yreversed=true)
image!(ax1, permutedims(A, (2,1)))
scr1 = display(GLMakie.Screen(title="Figure 1"), fig1)
GLFW.SetWindowPos(scr1.glscreen, 50, 50)

k = 120*2
use_single = true  # set to true for single precision

if use_single
    conv = Float32
else
    conv = Float64
end

# Convert RGB image to 2D matrix: m x (n*nc)
# Extract R, G, B channels and concatenate horizontally
R = conv.(red.(A)) * 255
G = conv.(green.(A)) * 255
B_ch = conv.(blue.(A)) * 255
A2 = hcat(R, G, B_ch)

t0 = time()
U, s, Vt = librla.svd_sketch(A2, k)
B2 = U * diagm(s) * Vt
println("Elapsed time: $(round(time() - t0, digits=3))s")

# Reshape back to RGB
R_rec = clamp.(B2[:, 1:n], 0, 255) / 255
G_rec = clamp.(B2[:, n+1:2n], 0, 255) / 255
B_rec = clamp.(B2[:, 2n+1:3n], 0, 255) / 255
img_rec = RGB.(R_rec, G_rec, B_rec)

fig2 = Figure()
ax2 = GLMakie.Axis(fig2[1,1], title="Rank-$k randomized SVD", aspect=DataAspect(), yreversed=true)
image!(ax2, permutedims(img_rec, (2,1)))
scr2 = display(GLMakie.Screen(title="Figure 2"), fig2)
GLFW.SetWindowPos(scr2.glscreen, 100, 100)

rel_error = norm(A2 - B2) / norm(A2)
println("Basic SVD relative error: $(round(rel_error, sigdigits=6))")


t0 = time()
U, s, Vt = librla.svd_sketch(A2, k, power_iter=1, extra_samples=div(k, 4))
B2 = U * diagm(s) * Vt
println("Elapsed time: $(round(time() - t0, digits=3))s")

# Reshape back to RGB
R_rec = clamp.(B2[:, 1:n], 0, 255) / 255
G_rec = clamp.(B2[:, n+1:2n], 0, 255) / 255
B_rec = clamp.(B2[:, 2n+1:3n], 0, 255) / 255
img_rec = RGB.(R_rec, G_rec, B_rec)

fig3 = Figure()
ax3 = GLMakie.Axis(fig3[1,1], title="Rank-$k randomized SVD (power iterations, extra sampling)", aspect=DataAspect(), yreversed=true)
image!(ax3, permutedims(img_rec, (2,1)))
scr3 = display(GLMakie.Screen(title="Figure 3"), fig3)
GLFW.SetWindowPos(scr3.glscreen, 150, 150)

rel_error = norm(A2 - B2) / norm(A2)
println("Power iter SVD relative error: $(round(rel_error, sigdigits=6))")

