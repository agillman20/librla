%==========================================================================
% demo_utils.m - Shared utilities for librla demos
%
% Matrix generators and helper functions used across all demos.
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: NIST-PD
% Assisted by: Claude Code (Anthropic)
%==========================================================================

classdef demo_utils
    methods(Static)

        %==================================================================
        % Matrix Generators
        %==================================================================

        function H = hilbert(m, n)
            %HILBERT Generate m x n Hilbert matrix.
            %
            %   H = demo_utils.hilbert(m, n)
            %
            %   The Hilbert matrix is severely ill-conditioned, with entries
            %   H(i,j) = 1/(i+j-1). Useful for testing numerical stability.
            if nargin < 2
                n = m;
            end
            [i, j] = ndgrid(1:m, 1:n);
            H = 1.0 ./ (i + j - 1);
        end

        function K = kahan(m, n, theta, pert)
            %KAHAN Generate m x n Kahan matrix.
            %
            %   K = demo_utils.kahan(m)
            %   K = demo_utils.kahan(m, n)
            %   K = demo_utils.kahan(m, n, theta)
            %   K = demo_utils.kahan(m, n, theta, pert)
            %
            %   Upper triangular matrix with exponentially decaying rows.
            %   Classic test for QR factorization algorithms.
            if nargin < 2 || isempty(n), n = m; end
            if nargin < 3, theta = 1.2; end
            if nargin < 4, pert = 25; end

            s = sin(theta);
            c = cos(theta);
            ep = eps;
            r = min(m, n);

            K = zeros(m, n);

            % Set diagonal
            for i = 1:r
                K(i, i) = 1.0;
            end

            % Set upper triangular part
            for i = 1:m
                for j = i+1:n
                    K(i, j) = -c;
                end
            end

            % Scale rows by s^(i-1)
            for i = 1:m
                K(i, :) = K(i, :) * (s^(i-1));
            end

            % Add diagonal perturbation
            for i = 1:r
                K(i, i) = K(i, i) + pert * ep * (r - i + 1);
            end
        end

        function [A, s] = lowrank(m, n, k, decay, gap)
            %LOWRANK Generate m x n matrix with controlled rank-k structure.
            %
            %   [A, s] = demo_utils.lowrank(m, n, k)
            %   [A, s] = demo_utils.lowrank(m, n, k, decay, gap)
            %
            %   Creates a matrix where the first k singular values are
            %   well-separated from the remaining ones.
            if nargin < 4, decay = 'exponential'; end
            if nargin < 5, gap = 100.0; end

            r = min(m, n);

            if strcmp(decay, 'exponential')
                s = [logspace(0, -2, k), logspace(-2, -10, r-k) / gap];
            elseif strcmp(decay, 'polynomial')
                s = [1.0 ./ (1:k).^2, 1.0 ./ ((k+1):r).^2 / gap];
            elseif strcmp(decay, 'step')
                s = [ones(1, k), ones(1, r-k) / gap];
            else
                error('Unknown decay type: %s', decay);
            end
            s = s(:);

            [U, ~] = qr(randn(m, r), 0);
            [V, ~] = qr(randn(n, r), 0);

            A = U * diag(s) * V';
        end

        function A = random_matrix(m, n, seed)
            %RANDOM_MATRIX Generate m x n random Gaussian matrix.
            %
            %   A = demo_utils.random_matrix(m, n)
            %   A = demo_utils.random_matrix(m, n, seed)
            if nargin >= 3 && ~isempty(seed)
                rng(seed);
            end
            A = randn(m, n);
        end

        %==================================================================
        % Error Computation
        %==================================================================

        function err = id_error(A, k, piv, T)
            %ID_ERROR Compute relative ID reconstruction error.
            %
            %   err = demo_utils.id_error(A, k, piv, T)
            %
            %   The ID approximation is: A(:, piv(k+1:end)) = A(:, piv(1:k)) * T
            A_basis = A(:, piv(1:k));
            A_skel = A(:, piv(k+1:end));
            err = norm(A_skel - A_basis * T, 'fro') / norm(A, 'fro');
        end

        function err = svd_error(A, U, s, V)
            %SVD_ERROR Compute relative SVD reconstruction error.
            %
            %   err = demo_utils.svd_error(A, U, s, V)
            %
            %   Note: MATLAB convention - V is not transposed.
            A_approx = U * diag(s) * V';
            err = norm(A - A_approx, 'fro') / norm(A, 'fro');
        end

        %==================================================================
        % Display Helpers
        %==================================================================

        function print_header(title)
            %PRINT_HEADER Print formatted section header.
            fprintf('======================================================================\n');
            fprintf('%s\n', title);
            fprintf('======================================================================\n');
        end

        function print_subheader(title)
            %PRINT_SUBHEADER Print formatted subsection header.
            fprintf('\n%s\n', title);
            fprintf('----------------------------------------------------------------------\n');
        end

    end
end
