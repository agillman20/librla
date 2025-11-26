%TEST_IMAGE_ID  Compare low-rank image compression methods from librla.
%
%  Description
%  -----------
%    This script compares six low-rank approximation methods on RGB images:
%
%      1. Randomized SVD (svd_sketch)
%      2. Interpolative Decomposition (id_sketch)
%      3. Randomized SVD with oversampling and power iteration
%      4. ID with oversampling and power iteration
%      5. QR with column pivoting (qr_sketch)
%
%    The image is reshaped to a 2D matrix (m x 3n) for processing.
%
%    ID selects k skeleton columns and expresses remaining columns as
%    linear combinations: A(:, piv(k+1:end)) ≈ A(:, piv(1:k)) * T
%
%    Two-sided ID selects both skeleton columns and rows, storing only
%    a small core submatrix C = A(I,J) plus interpolation matrices.
%
%    Set use_single=true to run in single precision (faster, less memory).
%
%  Requirements
%  ------------
%    * librla.m in the MATLAB path
%    * Image file (see below)
%    * Octave users need the image package (loaded automatically)
%
%  See also: librla.id_sketch, librla.svd_sketch, librla.qr_sketch, imread
%
% ----------------------------------------------------------------------

if exist('OCTAVE_VERSION', 'builtin')
    pkg load image
end

%%A = imread('pexels-flickr-149387.jpg');
%%A = imread('pexels-anniroenkae-4793404.jpg');
A = imread('x1d-II-sample-02.jpg');

%%A = permute(A, [2 1 3]);

[m, n, nc] = size(A);
fprintf('Image size: %d x %d x %d\n', m, n, nc);

figure(1)
imagesc(A)
axis image
title('Original (RGB)')

k = 60;
use_single = false;  % set to true for single precision

if use_single
    conv = @single;
else
    conv = @double;
end

% Reshape RGB image to 2D matrix: m x (n*nc)
A2 = reshape(conv(A), m, n*nc);

%% ========================================================================
% Method 1: Randomized SVD for comparison
%% ========================================================================
fprintf('\n--- Randomized SVD ---\n');
tic;
[U, s, V] = librla.svd_sketch(A2, k);
B2_svd = U(:,1:k) * diag(s(1:k)) * V(:,1:k)';
toc

B_svd = reshape(B2_svd, m, n, nc);

figure(2)
imagesc(uint8(B_svd))
axis image
title(sprintf('Rank-%d randomized SVD', k))

rel_error_svd = norm(A2 - B2_svd, 'fro') / norm(A2, 'fro');
fprintf('SVD relative error: %.6e\n', rel_error_svd);

%% ========================================================================
% Method 2: Interpolative Decomposition (randomized)
%% ========================================================================
fprintf('\n--- Interpolative Decomposition (id_sketch) ---\n');
tic;
[k_id, piv, T] = librla.id_sketch(A2, k);
toc

fprintf('ID rank: %d\n', k_id);

% Reconstruct: skeleton columns + interpolated columns
% A(:, piv(1:k)) are skeleton columns (kept exactly)
% A(:, piv(k+1:end)) ≈ A(:, piv(1:k)) * T

% Build reconstruction in permuted order, then unpermute
skeleton = A2(:, piv(1:k_id));
interpolated = skeleton * T;

B2_id_perm = [skeleton, interpolated];  % columns in permuted order

% Unpermute to original column order
B2_id = zeros(size(A2), class(A2));
B2_id(:, piv(1:k_id)) = skeleton;
B2_id(:, piv(k_id+1:end)) = interpolated;

B_id = reshape(B2_id, m, n, nc);

figure(3)
imagesc(uint8(B_id))
axis image
title(sprintf('Rank-%d interpolative decomposition', k_id))

rel_error_id = norm(A2 - B2_id, 'fro') / norm(A2, 'fro');
fprintf('ID relative error: %.6e\n', rel_error_id);

%% ========================================================================
% Method 3: Randomized SVD with oversampling and power iteration
%% ========================================================================
extra = floor(0.5 * k);  % 50% oversampling
piter = 2;               % power iterations
fprintf('\n--- Randomized SVD (extra_samples=%d, power_iter=%d) ---\n', extra, piter);
tic;
[U3, s3, V3] = librla.svd_sketch(A2, k, 'extra_samples', extra, 'power_iter', piter);
B2_svd2 = U3(:,1:k) * diag(s3(1:k)) * V3(:,1:k)';
toc

B_svd2 = reshape(B2_svd2, m, n, nc);

figure(4)
imagesc(uint8(B_svd2))
axis image
title(sprintf('Rank-%d SVD (extra=%d, piter=%d)', k, extra, piter))

rel_error_svd2 = norm(A2 - B2_svd2, 'fro') / norm(A2, 'fro');
fprintf('SVD (oversampled+piter) relative error: %.6e\n', rel_error_svd2);

%% ========================================================================
% Method 4: Interpolative Decomposition with oversampling and power iteration
%% ========================================================================
fprintf('\n--- ID (extra_samples=%d, power_iter=%d) ---\n', extra, piter);
tic;
[k_id2, piv2, T2] = librla.id_sketch(A2, k, 'extra_samples', extra, 'power_iter', piter);
toc

fprintf('ID rank: %d\n', k_id2);

% Reconstruct
skeleton2 = A2(:, piv2(1:k_id2));
interpolated2 = skeleton2 * T2;

B2_id2 = zeros(size(A2), class(A2));
B2_id2(:, piv2(1:k_id2)) = skeleton2;
B2_id2(:, piv2(k_id2+1:end)) = interpolated2;

B_id2 = reshape(B2_id2, m, n, nc);

figure(5)
imagesc(uint8(B_id2))
axis image
title(sprintf('Rank-%d ID (extra=%d, piter=%d)', k_id2, extra, piter))

rel_error_id2 = norm(A2 - B2_id2, 'fro') / norm(A2, 'fro');
fprintf('ID (oversampled+piter) relative error: %.6e\n', rel_error_id2);

%% ========================================================================
% Method 5: QR with column pivoting (via qr_sketch)
%% ========================================================================
fprintf('\n--- QR (extra_samples=%d, power_iter=%d) ---\n', extra, piter);
tic;
[Q_qr, R_qr, p_qr] = librla.qr_sketch(A2, k, 'extra_samples', extra, 'power_iter', piter);
toc

k_qr = size(Q_qr, 2);
fprintf('QR rank: %d\n', k_qr);

% Reconstruct: A(:, p) ≈ Q * R, so unpermute columns
B2_qr_perm = Q_qr * R_qr;  % columns in permuted order
B2_qr = zeros(size(A2), class(A2));
B2_qr(:, p_qr) = B2_qr_perm;

B_qr = reshape(B2_qr, m, n, nc);

figure(6)
imagesc(uint8(B_qr))
axis image
title(sprintf('Rank-%d QR (extra=%d, piter=%d)', k_qr, extra, piter))

rel_error_qr = norm(A2 - B2_qr, 'fro') / norm(A2, 'fro');
fprintf('QR relative error: %.6e\n', rel_error_qr);

%% ========================================================================
% Summary
%% ========================================================================
fprintf('\n=== Summary ===\n');
fprintf('Method                              Rank    Relative Error\n');
fprintf('-----------------------------------------------------------\n');
fprintf('Randomized SVD                      %4d    %.6e\n', k, rel_error_svd);
fprintf('ID (randomized)                     %4d    %.6e\n', k_id, rel_error_id);
fprintf('SVD (extra=%d, piter=%d)            %4d    %.6e\n', extra, piter, k, rel_error_svd2);
fprintf('ID  (extra=%d, piter=%d)            %4d    %.6e\n', extra, piter, k_id2, rel_error_id2);
fprintf('QR  (extra=%d, piter=%d)            %4d    %.6e\n', extra, piter, k_qr, rel_error_qr);
