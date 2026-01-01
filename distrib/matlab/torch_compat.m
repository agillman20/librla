classdef torch_compat
% TORCH_COMPAT - PyTorch-compatible wrappers for librla
%
% Thin wrappers providing torch.svd_lowrank and torch.pca_lowrank
% compatible interfaces using librla's randomized algorithms.
%
% Usage:
%   [U, s, V] = torch_compat.svd_lowrank(A, q, niter, M);
%   [U, s, V] = torch_compat.pca_lowrank(A, q, center, niter);
%
% Note: q is the oversampled rank (sketch size), not the final rank.
% User is responsible for choosing q = k + oversampling where k is
% the target rank. Typical oversampling is 5-10.
%
% Reference: Halko et al., "Finding structure with randomness" (2009)
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: TBD
% Version: 0.1.0
% Date: TBD
% Assisted by: Claude Code (Anthropic)

methods (Static)

function [U, s, V] = svd_lowrank(A, q, niter, M)
% SVD_LOWRANK - PyTorch-compatible randomized low-rank SVD
%
% Syntax:
%   [U, s, V] = torch_compat.svd_lowrank(A)
%   [U, s, V] = torch_compat.svd_lowrank(A, q)
%   [U, s, V] = torch_compat.svd_lowrank(A, q, niter)
%   [U, s, V] = torch_compat.svd_lowrank(A, q, niter, M)
%
% Input Arguments:
%   A     - Input matrix (m x n)
%   q     - Oversampled rank / sketch size (default: 6)
%   niter - Power iterations (default: 2)
%   M     - Matrix to subtract before decomposition (default: [])
%
% Output Arguments:
%   U - Left singular vectors (m x q)
%   s - Singular values (q x 1 vector)
%   V - Right singular vectors (n x q), NOT transposed
%
% Note: Unlike librla.svd_sketch which returns V (not transposed),
% this function matches PyTorch convention and also returns V (not transposed).

    % Default arguments
    if nargin < 2 || isempty(q)
        q = 6;
    end
    if nargin < 3 || isempty(niter)
        niter = 2;
    end
    if nargin < 4
        M = [];
    end

    % Subtract M if provided
    if ~isempty(M)
        A = A - M;
    end

    % Call librla.svd_sketch with rank mode
    % MATLAB librla returns V (not transposed), matching PyTorch convention
    [U, s, V] = librla.svd_sketch(A, q, 'power_iter', niter, 'extra_samples', 0);
end

function [U, s, V] = pca_lowrank(A, q, center, niter)
% PCA_LOWRANK - PyTorch-compatible randomized low-rank PCA
%
% Syntax:
%   [U, s, V] = torch_compat.pca_lowrank(A)
%   [U, s, V] = torch_compat.pca_lowrank(A, q)
%   [U, s, V] = torch_compat.pca_lowrank(A, q, center)
%   [U, s, V] = torch_compat.pca_lowrank(A, q, center, niter)
%
% Input Arguments:
%   A      - Input matrix (m x n) - m samples, n features
%   q      - Oversampled rank / sketch size (default: min(6, m, n))
%   center - Subtract column means (default: true)
%   niter  - Power iterations (default: 2)
%
% Output Arguments:
%   U - Left singular vectors (m x q)
%   s - Singular values (q x 1 vector)
%   V - Right singular vectors (n x q), NOT transposed
%
% Notes:
%   The relation to PCA:
%   - V columns are principal directions
%   - A * V(:, 1:k) projects data to first k principal components

    [m, n] = size(A);

    % Default arguments
    if nargin < 2 || isempty(q)
        q = min([6, m, n]);
    end
    if nargin < 3 || isempty(center)
        center = true;
    end
    if nargin < 4 || isempty(niter)
        niter = 2;
    end

    % Center data if requested
    if center
        A = A - mean(A, 1);
    end

    % Call librla.svd_sketch with rank mode
    [U, s, V] = librla.svd_sketch(A, q, 'power_iter', niter, 'extra_samples', 0);
end

end % methods (Static)
end % classdef
