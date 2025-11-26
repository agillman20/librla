%TEST_IMAGE_RGB  Demonstrate RGB image compression using truncated SVD.
%
%  Description
%  -----------
%    This script demonstrates low-rank image compression using the
%    randomized truncated SVD from librla. It loads an RGB image,
%    reshapes it to a 2D matrix (m x 3n) for SVD, and displays results.
%
%    Two compression methods are compared:
%      1. Basic truncated SVD with rank k
%      2. Truncated SVD with power iterations and extra samples for
%         improved accuracy
%
%    Set use_single=true to run in single precision (faster, less memory).
%
%  Requirements
%  ------------
%    * librla.m in the MATLAB path
%    * Image file: pexels-flickr-149387.jpg
%    * Octave users need the image package (loaded automatically)
%
%  See also: librla.svd_sketch, imread, imagesc, test_image
%
% ----------------------------------------------------------------------
% Image source: https://www.pexels.com/photo/silver-metal-round-gears-connected-to-each-other-149387/
% https://1.img-dpreview.com/files/p/sample_galleries/6044553814/1399808671.jpg
% ----------------------------------------------------------------------

if exist('OCTAVE_VERSION', 'builtin')
    pkg load image
end

%%A = imread('pexels-flickr-149387.jpg');
A = imread('pexels-anniroenkae-4793404.jpg');
%%A = imread('x1d-II-sample-02.jpg');
A = imread('b_29667.jpg');

%%A = permute(A,[2 1 3]);

[m, n, nc] = size(A);
fprintf('Image size: %d x %d x %d\n', m, n, nc);

figure(1)
imagesc(A)
axis image
title('Original (RGB)')

k = 120
use_single = false;  % set to true for single precision

if use_single
    conv = @single;
else
    conv = @double;
end

% Reshape RGB image to 2D matrix: m x (n*nc)
A2 = reshape(conv(A), m, n*nc);

tic;
[U,s,V] = librla.svd_sketch(A2, k);
B2 = U(:,1:k)*diag(s(1:k))*V(:,1:k)';
toc

% Reshape back to RGB
B = reshape(B2, m, n, nc);

figure(2)
imagesc(uint8(B))
axis image
title(sprintf('Rank-%d randomized SVD', k))

rel_error = norm(A2-B2,'fro')/norm(A2,'fro');
fprintf('Basic SVD relative error: %.6e\n', rel_error);


tic
[U,s,V] = librla.svd_sketch(A2, k, 'power_iter', 1, 'extra_samples', floor(.25*k));
B2 = U(:,1:k)*diag(s(1:k))*V(:,1:k)';
toc

% Reshape back to RGB
B = reshape(B2, m, n, nc);

figure(3)
imagesc(uint8(B))
axis image
title(sprintf('Rank-%d randomized SVD (power iterations, extra sampling)', k))

rel_error = norm(A2-B2,'fro')/norm(A2,'fro');
fprintf('Power iter SVD relative error: %.6e\n', rel_error);
