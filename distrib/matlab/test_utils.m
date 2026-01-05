%==========================================================================
% test_utils.m - Shared utilities for librla tests
%
% Matrix generators and helper functions used across all test files.
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: BSD-3-Clause AND NIST-PD
% Assisted by: Claude Code (Anthropic)
%==========================================================================

classdef test_utils
    methods(Static)

        %==================================================================
        % Matrix Generators (from make_mat)
        %==================================================================

        function X = make_mat(m, n, flag_type)
            %MAKE_MAT Generate test matrices for ID benchmarking.
            %
            %   X = test_utils.make_mat(m, n, flag_type)
            %
            %   Arguments:
            %     m         - Number of rows
            %     n         - Number of columns
            %     flag_type - Matrix type: 'cifar', 'mnist', 'gaussexp', 'gmm', 'snn'
            %
            %   Returns:
            %     X - Generated matrix (normalized by column)

            if strcmp(flag_type, 'cifar')
                if (n > m)
                    error('columns must be less than or equal to rows for CIFAR');
                end
                if m > 3072
                    error('too many rows! CIFAR-10 has 3072 features (32x32x3)');
                end
                MM = load('exampledata/cifar10.mat');
                A = MM.full_matrix.';
                X = A(1:m, randperm(60000, n));

            elseif strcmp(flag_type, 'mnist')
                if (n > m)
                    error('columns must be less than or equal to rows for MNIST');
                end
                if m > 784
                    error('too many rows! MNIST has 784 features (28x28)');
                end
                MM = load('exampledata/mnist_mat.mat');
                A = MM.mnist_mat.';
                X = A(1:m, randperm(60000, n));

            elseif strcmp(flag_type, 'gaussexp')
                X = test_utils.Matrix_Gaussian_exp(m);

            elseif strcmp(flag_type, 'gmm')
                X = test_utils.Matrix_GMM(n, m);

            elseif strcmp(flag_type, 'snn')
                X = test_utils.Matrix_SNN(n);

            else
                error('Unknown flag_type: %s. Valid types: cifar, mnist, gaussexp, gmm, snn', flag_type);
            end

            % Normalize by column
            col_norms = sqrt(sum(X.^2, 1));
            X = bsxfun(@rdivide, X, col_norms);
        end

        function A = Matrix_GMM(n, d)
            %MATRIX_GMM Generate Gaussian Mixture Model matrix.
            k = 100;
            m = floor(n / k);

            A = randn(n, d);

            for i = 1:min(k, d)
                I = 1 + (i-1)*m : i*m;
                A(I, i) = 10*i + A(I, i);
            end

            A = A';
        end

        function A = Matrix_Gaussian_exp(n)
            %MATRIX_GAUSSIAN_EXP Generate matrix with exponentially decaying SVs.
            m = 100;
            sv = nan(1, n);
            sv(1:m) = 1;
            sv(m+1:end) = 0.8.^(1:n-m);
            sv(sv < 1e-5) = 1e-5;

            [U, ~] = qr(randn(n));
            [V, ~] = qr(randn(n));
            A = U * diag(sv) * V';
        end

        function A = Matrix_SNN(n)
            %MATRIX_SNN Generate Sparse Neural Network matrix.
            m = 100;
            sv = nan(1, n);
            sv(1:m) = 10 ./ (1:m);
            sv(m+1:end) = 1 ./ (m+1:n);

            U = sprand(n, n, 0.1);
            V = sprand(n, n, 0.1);

            A = U * diag(sv) * V';
        end

        %==================================================================
        % Additional Matrix Generators (from demo_utils)
        %==================================================================

        function [A, s] = lowrank(m, n, k, decay, gap)
            %LOWRANK Generate m x n matrix with controlled rank-k structure.
            %
            %   [A, s] = test_utils.lowrank(m, n, k)
            %   [A, s] = test_utils.lowrank(m, n, k, decay, gap)
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
            %   A = test_utils.random_matrix(m, n)
            %   A = test_utils.random_matrix(m, n, seed)
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
            %   err = test_utils.id_error(A, k, piv, T)
            %
            %   The ID approximation is: A(:, piv(k+1:end)) = A(:, piv(1:k)) * T
            A_basis = A(:, piv(1:k));
            A_skel = A(:, piv(k+1:end));
            err = norm(A_skel - A_basis * T, 'fro') / norm(A, 'fro');
        end

        function err = svd_error(A, U, s, V)
            %SVD_ERROR Compute relative SVD reconstruction error.
            %
            %   err = test_utils.svd_error(A, U, s, V)
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
