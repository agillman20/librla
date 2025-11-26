#=
test_image_id.jl - Compare low-rank image compression methods from librla.

Description
-----------
This script compares low-rank approximation methods on RGB images:

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
* Packages: Images, FileIO, PyPlot

See also: librla.id_sketch, librla.svd_sketch, librla.qr_sketch
=#

using LinearAlgebra
using Images
using FileIO
using PyPlot
using Printf

# Include librla from parent directory
include(joinpath(@__DIR__, "..", "julia", "librla.jl"))
using .librla

# Load image
# A = load("pexels-flickr-149387.jpg")
# A = load("pexels-anniroenkae-4793404.jpg")
A = load("x1d-II-sample-02.jpg")

# A = permutedims(A, (2, 1))

# Get dimensions
m, n = size(A)
nc = 3  # RGB channels

println("Image size: $m x $n x $nc")

# Convert to array for display
A_arr = permutedims(channelview(A), (2, 3, 1))

figure(1)
imshow(A_arr)
title("Original (RGB)")

k = 60
use_single = false  # set to true for single precision

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
println("\n--- Randomized SVD ---")
t0 = time()
U, s, Vt = librla.svd_sketch(A2, k)
B2_svd = U * diagm(s) * Vt
@printf("Elapsed time: %.3fs\n", time() - t0)

# Reshape back to RGB
R_rec = clamp.(B2_svd[:, 1:n], 0, 255) / 255
G_rec = clamp.(B2_svd[:, n+1:2n], 0, 255) / 255
B_rec = clamp.(B2_svd[:, 2n+1:3n], 0, 255) / 255
img_arr = cat(R_rec, G_rec, B_rec, dims=3)

figure(2)
imshow(img_arr)
title("Rank-$k randomized SVD")

rel_error_svd = norm(A2 - B2_svd) / norm(A2)
@printf("SVD relative error: %.6e\n", rel_error_svd)

# ========================================================================
# Method 2: Interpolative Decomposition (randomized)
# ========================================================================
println("\n--- Interpolative Decomposition (id_sketch) ---")
t0 = time()
k_id, piv, T = librla.id_sketch(A2, k)
@printf("Elapsed time: %.3fs\n", time() - t0)

println("ID rank: $k_id")

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
img_arr = cat(R_rec, G_rec, B_rec, dims=3)

figure(3)
imshow(img_arr)
title("Rank-$k_id interpolative decomposition")

rel_error_id = norm(A2 - B2_id) / norm(A2)
@printf("ID relative error: %.6e\n", rel_error_id)

# ========================================================================
# Method 3: Randomized SVD with oversampling and power iteration
# ========================================================================
extra = div(k, 2)  # 50% oversampling
piter = 2          # power iterations
println("\n--- Randomized SVD (extra_samples=$extra, power_iter=$piter) ---")
t0 = time()
U3, s3, Vt3 = librla.svd_sketch(A2, k, extra_samples=extra, power_iter=piter)
B2_svd2 = U3 * diagm(s3) * Vt3
@printf("Elapsed time: %.3fs\n", time() - t0)

# Reshape back to RGB
R_rec = clamp.(B2_svd2[:, 1:n], 0, 255) / 255
G_rec = clamp.(B2_svd2[:, n+1:2n], 0, 255) / 255
B_rec = clamp.(B2_svd2[:, 2n+1:3n], 0, 255) / 255
img_arr = cat(R_rec, G_rec, B_rec, dims=3)

figure(4)
imshow(img_arr)
title("Rank-$k SVD (extra=$extra, piter=$piter)")

rel_error_svd2 = norm(A2 - B2_svd2) / norm(A2)
@printf("SVD (oversampled+piter) relative error: %.6e\n", rel_error_svd2)

# ========================================================================
# Method 4: Interpolative Decomposition with oversampling and power iteration
# ========================================================================
println("\n--- ID (extra_samples=$extra, power_iter=$piter) ---")
t0 = time()
k_id2, piv2, T2 = librla.id_sketch(A2, k, extra_samples=extra, power_iter=piter)
@printf("Elapsed time: %.3fs\n", time() - t0)

println("ID rank: $k_id2")

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
img_arr = cat(R_rec, G_rec, B_rec, dims=3)

figure(5)
imshow(img_arr)
title("Rank-$k_id2 ID (extra=$extra, piter=$piter)")

rel_error_id2 = norm(A2 - B2_id2) / norm(A2)
@printf("ID (oversampled+piter) relative error: %.6e\n", rel_error_id2)

# ========================================================================
# Method 5: QR with column pivoting (via qr_sketch)
# ========================================================================
println("\n--- QR (extra_samples=$extra, power_iter=$piter) ---")
t0 = time()
Q_qr, R_qr, p_qr = librla.qr_sketch(A2, k, extra_samples=extra, power_iter=piter)
@printf("Elapsed time: %.3fs\n", time() - t0)

k_qr = size(Q_qr, 2)
println("QR rank: $k_qr")

# Reconstruct: A[:, p] = Q * R, so unpermute columns
B2_qr_perm = Q_qr * R_qr  # columns in permuted order
B2_qr = zeros(conv, size(A2))
B2_qr[:, p_qr] = B2_qr_perm

# Reshape back to RGB
R_rec = clamp.(B2_qr[:, 1:n], 0, 255) / 255
G_rec = clamp.(B2_qr[:, n+1:2n], 0, 255) / 255
B_rec = clamp.(B2_qr[:, 2n+1:3n], 0, 255) / 255
img_arr = cat(R_rec, G_rec, B_rec, dims=3)

figure(6)
imshow(img_arr)
title("Rank-$k_qr QR (extra=$extra, piter=$piter)")

rel_error_qr = norm(A2 - B2_qr) / norm(A2)
@printf("QR relative error: %.6e\n", rel_error_qr)

# ========================================================================
# Summary
# ========================================================================
println("\n=== Summary ===")
println("Method                              Rank    Relative Error")
println("-----------------------------------------------------------")
@printf("Randomized SVD                      %4d    %.6e\n", k, rel_error_svd)
@printf("ID (randomized)                     %4d    %.6e\n", k_id, rel_error_id)
@printf("SVD (extra=%d, piter=%d)            %4d    %.6e\n", extra, piter, k, rel_error_svd2)
@printf("ID  (extra=%d, piter=%d)            %4d    %.6e\n", extra, piter, k_id2, rel_error_id2)
@printf("QR  (extra=%d, piter=%d)            %4d    %.6e\n", extra, piter, k_qr, rel_error_qr)

