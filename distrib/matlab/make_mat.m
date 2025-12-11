function X = make_mat(m,n,flag_type)
% MAKE_MAT - Generate test matrices for ID benchmarking
%
%   X = make_mat(m, n, flag_type)
%
% Generates various types of test matrices from:
% "Robust blockwise random pivoting: Fast and accurate adaptive
%  interpolative decomposition"
%
% Arguments:
%   m         - Number of rows
%   n         - Number of columns
%   flag_type - Matrix type: 'cifar', 'mnist', 'gaussexp', 'gmm', 'snn'
%
% Returns:
%   X - Generated matrix (normalized by column)
%
% Data Requirements:
%   cifar: Requires exampledata/cifar10.mat (or ../cifar/cifar-10-batches-mat/)
%   mnist: Requires exampledata/mnist_mat.mat
%   Other types generate synthetic matrices (no data files needed)
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: TBD
% Version: 1.0.0
% Date: TBD
% Assisted by: Claude Code (Anthropic)

if strcmp(flag_type,'cifar')
    if (n>m)
        disp('columns must be less than or equal to n')
        X = [];
        return
    end

    if m> 3072
        disp('too many rows!')
        X =[];
        return
    end
    MM = load('exampledata/cifar10.mat');
    A = MM.full_matrix.';
    X = A(1:m, randperm(60000, n));
elseif strcmp(flag_type,'mnist')
    if (n>m)
        disp('columns must be less than or equal to n')
        X = [];
        return
    end
    if m> 784
        disp('too many rows!')
        X =[];
        return
    end
    MM  = load('exampledata/mnist_mat.mat');
    A = MM.mnist_mat.';
    X = A(1:m, randperm(60000, n));
elseif strcmp(flag_type,'gaussexp')
    X = Matrix_Gaussian_exp(m);

elseif strcmp(flag_type,'gmm')
    X = Matrix_GMM(n, m);
    
elseif strcmp(flag_type,'snn')
    X = Matrix_SNN(n);
else
    error('Unknown flag_type: %s. Valid types: cifar, mnist, gaussexp, gmm, snn', flag_type);
end

% Normalize by column (compute norm of each column)
col_norms = sqrt(sum(X.^2, 1));
% Use bsxfun for Octave compatibility
X = bsxfun(@rdivide, X, col_norms);

return
end


function A = Matrix_GMM(n, d)

k = 100;
m = floor(n/k);

A = randn(n, d);

for i=1:min(k, d)  % Only modify columns that exist
    I = 1+(i-1)*m:i*m;
    A(I, i) = 10*i + A(I, i);
end

A = A';
return
end



function A = Matrix_Gaussian_exp(n)

% singular values decay fast
%sv = 1e-16 .^ ((0:n-1)/(n-1));

m = 100;
sv = nan(1,n);
sv(1:m) = 1;
sv(m+1:end) = 0.8.^(1:n-m);
sv(sv<1e-5)=1e-5;

[U,~] = qr(randn(n));
[V,~] = qr(randn(n));
A = U*diag(sv)*V';

return
end


function A = Matrix_SNN(n)

m = 100;
sv = nan(1,n);
sv(1:m) = 10 ./ (1:m);
sv(m+1:end) = 1 ./ (m+1:n);

U = sprand(n, n, 0.1);
V = sprand(n, n, 0.1);

A = U*diag(sv)*V';
return
end