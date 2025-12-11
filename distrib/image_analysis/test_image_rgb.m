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
%  Image source: https://www.pexels.com/photo/silver-metal-round-gears-connected-to-each-other-149387/
%  https://1.img-dpreview.com/files/p/sample_galleries/6044553814/1399808671.jpg
%
%  Author: Adrianna Gillman, Zydrunas Gimbutas
%  SPDX-License-Identifier: TBD
%  Version: 1.0.0
%  Date: TBD
%  Assisted by: Claude Code (Anthropic)
% ----------------------------------------------------------------------

if exist('OCTAVE_VERSION', 'builtin')
    pkg load image
end

%%image_file = 'pexels-flickr-149387.jpg';
%%image_file = 'pexels-anniroenkae-4793404.jpg';
%%image_file = 'b_29667.jpg';
image_file = 'pexels-andre-ulysses-de-salis-2100065-7824822.jpg';
A = imread(image_file);

%%A = permute(A,[2 1 3]);

[m, n, nc] = size(A);
fprintf('Image: %s, size: %d x %d x %d\n', image_file, m, n, nc);

figure(1)
imagesc(A)
axis image
title('Original (RGB)')

k = 120*2 
use_single = true;  % set to true for single precision

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
elapsed1 = toc;

% Reshape back to RGB
B = reshape(B2, m, n, nc);

figure(2)
imagesc(uint8(B))
axis image
title(strrep(sprintf('Rank-%d svd_sketch', k),'_','\_'))

rel_error1 = norm(A2-B2,'fro')/norm(A2,'fro');
fprintf('svd_sketch(k=%d): %.3fs, error %.6e\n', k, elapsed1, rel_error1);


extra = floor(.25*k);
piter = 1;
tic
[U,s,V] = librla.svd_sketch(A2, k, 'power_iter', piter, 'extra_samples', extra);
B2 = U(:,1:k)*diag(s(1:k))*V(:,1:k)';
elapsed2 = toc;

% Reshape back to RGB
B = reshape(B2, m, n, nc);

figure(3)
imagesc(uint8(B))
axis image
title(strrep(sprintf('Rank-%d svd_sketch (extra_samples=%d, power_iter=%d)', k, extra, piter),'_','\_'))

rel_error2 = norm(A2-B2,'fro')/norm(A2,'fro');
fprintf('svd_sketch(k=%d, extra_samples=%d, power_iter=%d): %.3fs, error %.6e\n', k, extra, piter, elapsed2, rel_error2);

fprintf('\n%-40s %4s    %s\n', 'Method', 'Rank', 'Error');
fprintf('%s\n', repmat('-', 1, 55));
fprintf('%-40s %4d    %.6e\n', sprintf('svd_sketch(k=%d)', k), k, rel_error1);
fprintf('%-40s %4d    %.6e\n', sprintf('svd_sketch(k=%d, extra, power)', k), k, rel_error2);
