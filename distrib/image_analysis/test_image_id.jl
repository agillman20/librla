#=
test_image_id.jl - Compare low-rank image compression methods from librla.

Description
-----------
This script compares low-rank factorization methods on RGB images:

  1. Randomized SVD (svd_sketch)
  2. Interpolative Decomposition (id_sketch)
  3. Randomized SVD with oversampling and power iteration
  4. ID with oversampling and power iteration
  5. QR with column pivoting (qr_sketch)

The image is reshaped to a 2D matrix (m x 3n) for processing.

ID selects k skeleton columns and expresses remaining columns as
linear combinations: A[:, piv[k+1:end]] = A[:, piv[1:k]] * T

Set use_single=true to run in single precision (faster, less memory).

Requirements
------------
* librla.jl in the path
* Image file (see below)
* Packages: Images, FileIO, GLMakie

See also: librla.id_sketch, librla.svd_sketch, librla.qr_sketch

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.0.1
Date: April 22, 2026
Assisted by: Claude Code (Anthropic)
=#

using LinearAlgebra
using Images
using FileIO
using GLMakie
using GLFW
using Printf

# Include librla from parent directory
include(joinpath(@__DIR__, "..", "julia", "librla.jl"))
using .librla

# Load image
# image_file = "pexels-flickr-149387.jpg"
image_file = "pexels-anniroenkae-4793404.jpg"
A = load(image_file)

# A = permutedims(A, (2, 1))

# Get dimensions
m, n = size(A)
nc = 3  # RGB channels

println("Image: $image_file, size: $m x $n x $nc")

fig1 = Figure()
ax1 = GLMakie.Axis(fig1[1,1], title="Original (RGB)", aspect=DataAspect(), yreversed=true)
image!(ax1, permutedims(A, (2,1)))
scr1 = display(GLMakie.Screen(title="Figure 1"), fig1)
GLFW.SetWindowPos(scr1.glscreen, 50, 50)

k = 60*2
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

# ========================================================================
# Method 1: Randomized SVD for comparison
# ========================================================================
t0 = time()
U, s, Vt = librla.svd_sketch(A2, k)
B2_svd = U * diagm(s) * Vt
elapsed_svd = time() - t0

# Reshape back to RGB
R_rec = clamp.(B2_svd[:, 1:n], 0, 255) / 255
G_rec = clamp.(B2_svd[:, n+1:2n], 0, 255) / 255
B_rec = clamp.(B2_svd[:, 2n+1:3n], 0, 255) / 255
img_rec = RGB.(R_rec, G_rec, B_rec)

fig2 = Figure()
ax2 = GLMakie.Axis(fig2[1,1], title="Rank-$k svd_sketch", aspect=DataAspect(), yreversed=true)
image!(ax2, permutedims(img_rec, (2,1)))
scr2 = display(GLMakie.Screen(title="Figure 2"), fig2)
GLFW.SetWindowPos(scr2.glscreen, 100, 100)

rel_error_svd = norm(A2 - B2_svd) / norm(A2)
@printf("svd_sketch(k=%d): %.3fs, error %.6e\n", k, elapsed_svd, rel_error_svd)

# ========================================================================
# Method 2: Interpolative Decomposition (randomized)
# ========================================================================
t0 = time()
k_id, piv, T = librla.id_sketch(A2, k, method="lstsq")
elapsed_id = time() - t0

# Reconstruct: skeleton columns + interpolated columns
# A[:, piv[1:k]] are skeleton columns (kept exactly)
# A[:, piv[k+1:end]] = A[:, piv[1:k]] * T

# Build reconstruction
skeleton = A2[:, piv[1:k_id]]
interpolated = skeleton * T

# Unpermute to original column order
B2_id = zeros(conv, size(A2))
B2_id[:, piv[1:k_id]] = skeleton
B2_id[:, piv[k_id+1:end]] = interpolated

# Reshape back to RGB
R_rec = clamp.(B2_id[:, 1:n], 0, 255) / 255
G_rec = clamp.(B2_id[:, n+1:2n], 0, 255) / 255
B_rec = clamp.(B2_id[:, 2n+1:3n], 0, 255) / 255
img_rec = RGB.(R_rec, G_rec, B_rec)

fig3 = Figure()
ax3 = GLMakie.Axis(fig3[1,1], title="Rank-$k_id id_sketch", aspect=DataAspect(), yreversed=true)
image!(ax3, permutedims(img_rec, (2,1)))
scr3 = display(GLMakie.Screen(title="Figure 3"), fig3)
GLFW.SetWindowPos(scr3.glscreen, 150, 150)

rel_error_id = norm(A2 - B2_id) / norm(A2)
@printf("id_sketch(k=%d): %.3fs, error %.6e\n", k_id, elapsed_id, rel_error_id)

# ========================================================================
# Method 3: Randomized SVD with oversampling and power iteration
# ========================================================================
extra = div(k, 2)  # 50% oversampling
piter = 4          # power iterations
t0 = time()
U3, s3, Vt3 = librla.svd_sketch(A2, k, extra_samples=extra, power_iter=piter)
B2_svd2 = U3 * diagm(s3) * Vt3
elapsed_svd2 = time() - t0

# Reshape back to RGB
R_rec = clamp.(B2_svd2[:, 1:n], 0, 255) / 255
G_rec = clamp.(B2_svd2[:, n+1:2n], 0, 255) / 255
B_rec = clamp.(B2_svd2[:, 2n+1:3n], 0, 255) / 255
img_rec = RGB.(R_rec, G_rec, B_rec)

fig4 = Figure()
ax4 = GLMakie.Axis(fig4[1,1], title="Rank-$k svd_sketch (extra_samples=$extra, power_iter=$piter)", aspect=DataAspect(), yreversed=true)
image!(ax4, permutedims(img_rec, (2,1)))
scr4 = display(GLMakie.Screen(title="Figure 4"), fig4)
GLFW.SetWindowPos(scr4.glscreen, 200, 200)

rel_error_svd2 = norm(A2 - B2_svd2) / norm(A2)
@printf("svd_sketch(k=%d, extra_samples=%d, power_iter=%d): %.3fs, error %.6e\n", k, extra, piter, elapsed_svd2, rel_error_svd2)

# ========================================================================
# Method 4: Interpolative Decomposition with oversampling and power iteration
# ========================================================================
t0 = time()
k_id2, piv2, T2 = librla.id_sketch(A2, k, extra_samples=extra, power_iter=piter, method="lstsq")
elapsed_id2 = time() - t0

# Reconstruct
skeleton2 = A2[:, piv2[1:k_id2]]
interpolated2 = skeleton2 * T2

B2_id2 = zeros(conv, size(A2))
B2_id2[:, piv2[1:k_id2]] = skeleton2
B2_id2[:, piv2[k_id2+1:end]] = interpolated2

# Reshape back to RGB
R_rec = clamp.(B2_id2[:, 1:n], 0, 255) / 255
G_rec = clamp.(B2_id2[:, n+1:2n], 0, 255) / 255
B_rec = clamp.(B2_id2[:, 2n+1:3n], 0, 255) / 255
img_rec = RGB.(R_rec, G_rec, B_rec)

fig5 = Figure()
ax5 = GLMakie.Axis(fig5[1,1], title="Rank-$k_id2 id_sketch (extra_samples=$extra, power_iter=$piter)", aspect=DataAspect(), yreversed=true)
image!(ax5, permutedims(img_rec, (2,1)))
scr5 = display(GLMakie.Screen(title="Figure 5"), fig5)
GLFW.SetWindowPos(scr5.glscreen, 250, 250)

rel_error_id2 = norm(A2 - B2_id2) / norm(A2)
@printf("id_sketch(k=%d, extra_samples=%d, power_iter=%d): %.3fs, error %.6e\n", k_id2, extra, piter, elapsed_id2, rel_error_id2)

# ========================================================================
# Method 5: QR with column pivoting (via qr_sketch)
# ========================================================================
t0 = time()
Q_qr, R_qr, p_qr = librla.qr_sketch(A2, k, extra_samples=extra, power_iter=piter)
elapsed_qr = time() - t0

k_qr = size(Q_qr, 2)

# Reconstruct: A[:, p] = Q * R, so unpermute columns
B2_qr_perm = Q_qr * R_qr  # columns in permuted order
B2_qr = zeros(conv, size(A2))
B2_qr[:, p_qr] = B2_qr_perm

# Reshape back to RGB
R_rec = clamp.(B2_qr[:, 1:n], 0, 255) / 255
G_rec = clamp.(B2_qr[:, n+1:2n], 0, 255) / 255
B_rec = clamp.(B2_qr[:, 2n+1:3n], 0, 255) / 255
img_rec = RGB.(R_rec, G_rec, B_rec)

fig6 = Figure()
ax6 = GLMakie.Axis(fig6[1,1], title="Rank-$k_qr qr_sketch (extra_samples=$extra, power_iter=$piter)", aspect=DataAspect(), yreversed=true)
image!(ax6, permutedims(img_rec, (2,1)))
scr6 = display(GLMakie.Screen(title="Figure 6"), fig6)
GLFW.SetWindowPos(scr6.glscreen, 300, 300)

rel_error_qr = norm(A2 - B2_qr) / norm(A2)
@printf("qr_sketch(k=%d, extra_samples=%d, power_iter=%d): %.3fs, error %.6e\n", k_qr, extra, piter, elapsed_qr, rel_error_qr)

# ========================================================================
# Summary
# ========================================================================
println()
@printf("%-40s %4s    %s\n", "Method", "Rank", "Error")
println("-" ^ 55)
@printf("%-40s %4d    %.6e\n", "svd_sketch(k=$k)", k, rel_error_svd)
@printf("%-40s %4d    %.6e\n", "id_sketch(k=$k_id)", k_id, rel_error_id)
@printf("%-40s %4d    %.6e\n", "svd_sketch(k=$k, extra, power)", k, rel_error_svd2)
@printf("%-40s %4d    %.6e\n", "id_sketch(k=$k_id2, extra, power)", k_id2, rel_error_id2)
@printf("%-40s %4d    %.6e\n", "qr_sketch(k=$k_qr, extra, power)", k_qr, rel_error_qr)

