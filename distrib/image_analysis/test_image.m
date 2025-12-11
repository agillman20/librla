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
%  Image source: https://www.pexels.com/photo/silver-metal-round-gears-connected-to-each-other-149387/
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

image_file = 'pexels-flickr-149387.jpg';
%%image_file = 'pexels-anniroenkae-4793404.jpg';
%%image_file = 'hello_world.png';
%%image_file = 'lorem_ipsum.png';

A = imread(image_file);

%%A = permute(A,[2 1 3]);

if ndims(A) == 3 && size(A, 3) == 3
    A = rgb2gray(A);
end
fprintf('Image: %s, size: %d x %d\n', image_file, size(A, 1), size(A, 2));

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
elapsed1 = toc;

figure(2)
B = max(0, min(255, B));  % clip to valid range
imagesc(B)
colormap('gray')
colorbar
axis image
title(strrep(sprintf('Rank-%d svd_sketch', k),'_','\_'))

rel_error1 = norm(conv(A)-conv(B),'fro')/norm(conv(A),'fro');
fprintf('svd_sketch(k=%d): %.3fs, error %.6e\n', k, elapsed1, rel_error1);


extra = floor(.25*k);
piter = 1;
tic
[U,s,V] = librla.svd_sketch(conv(A),k,'power_iter',piter,'extra_samples',extra);
B = U(:,1:k)*diag(s(1:k))*V(:,1:k)';
elapsed2 = toc;

figure(3)
B = max(0, min(255, B));  % clip to valid range
imagesc(B)
colormap('gray')
colorbar
axis image
title(strrep(sprintf('Rank-%d svd_sketch (extra_samples=%d, power_iter=%d)', k, extra, piter),'_','\_'))

rel_error2 = norm(conv(A)-conv(B),'fro')/norm(conv(A),'fro');
fprintf('svd_sketch(k=%d, extra_samples=%d, power_iter=%d): %.3fs, error %.6e\n', k, extra, piter, elapsed2, rel_error2);

fprintf('\n%-40s %4s    %s\n', 'Method', 'Rank', 'Error');
fprintf('%s\n', repmat('-', 1, 55));
fprintf('%-40s %4d    %.6e\n', sprintf('svd_sketch(k=%d)', k), k, rel_error1);
fprintf('%-40s %4d    %.6e\n', sprintf('svd_sketch(k=%d, extra, power)', k), k, rel_error2);
