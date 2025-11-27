%TEST_IMAGE  Demonstrate image compression using truncated SVD.
%
%  Description
%  -----------
%    This script demonstrates low-rank image compression using the
%    randomized truncated SVD from librla. It loads a grayscale image,
%    computes a rank-k approximation, and displays the results.
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
%  See also: librla.svd_sketch, imread, imagesc
%
% ----------------------------------------------------------------------
% Image source: https://www.pexels.com/photo/silver-metal-round-gears-connected-to-each-other-149387/
% ----------------------------------------------------------------------

if exist('OCTAVE_VERSION', 'builtin')
    pkg load image
end

A = imread('pexels-flickr-149387.jpg');
%%A = imread('pexels-anniroenkae-4793404.jpg');

%%A = permute(A,[2 1 3]);

if ndims(A) == 3 && size(A, 3) == 3
    A = rgb2gray(A);
end
size(A)

figure(1)
imagesc(A)
colormap('gray')
colorbar
axis image
title('Original (grayscale)')

k = 60*2
use_single = true;  % set to true for single precision

if use_single
    conv = @single;
else
    conv = @double;
end

tic;
[U,s,V] = librla.svd_sketch(conv(A),k);
B = U(:,1:k)*diag(s(1:k))*V(:,1:k)';
toc

figure(2)
B = max(0, min(255, B));  % clip to valid range
imagesc(B)
colormap('gray')
colorbar
axis image
title(sprintf('Rank-%d SVD approximation', k))

rel_error = norm(conv(A)-conv(B),'fro')/norm(conv(A),'fro')


tic
[U,s,V] = librla.svd_sketch(conv(A),k,'power_iter',1,'extra_samples',floor(.25*k));
B = U(:,1:k)*diag(s(1:k))*V(:,1:k)';
toc

figure(3)
B = max(0, min(255, B));  % clip to valid range
imagesc(B)
colormap('gray')
colorbar
axis image
title(sprintf('Rank-%d SVD (power iterations, extra sampling)', k))

rel_error = norm(conv(A)-conv(B),'fro')/norm(conv(A),'fro')
