function K = kahan(m, n, varargin)
% KAHAN  Generate Kahan matrix
%
%   K = kahan(m)
%   K = kahan(m, n)
%   K = kahan(m, n, theta)
%   K = kahan(m, n, theta, pert)
%
% Kahan's matrix is a classic test matrix for numerical stability.
% It's upper triangular with controlled condition number.
%
% Arguments:
%   m     - Number of rows
%   n     - Number of columns (default: m)
%   theta - Angle parameter in radians (default: 1.2)
%           Controls the condition number via cos(theta)
%           Smaller theta -> better conditioned
%           Larger theta -> worse conditioned
%   pert  - Perturbation parameter for diagonal entries (default: 25)
%           Standard form uses pert = 25 for numerical stability
%           Setting pert = 0 gives no diagonal perturbation
%
% Returns:
%   K - Kahan matrix (m x n)
%
% Matrix Structure:
%   K(i,i) = s^(i-1) + pert*eps*(n-i+1)  for i = 1,...,n (diagonal)
%   K(i,j) = -c * s^(i-1)                for i < j       (upper triangle)
%   K(i,j) = 0                           for i > j       (lower triangle)
%
%   where s = sin(theta), c = cos(theta), and eps is machine epsilon.
%
% The diagonal perturbation (pert*eps*(n-i+1)) ensures QR factorization
% with column pivoting does not interchange columns in the presence of
% rounding errors. The default pert=25 ensures no interchanges up to
% N=90 in IEEE arithmetic.
%
% The condition number is approximately 1/cos(theta)^n, so it grows
% exponentially with n and theta.
%
% Examples:
%   % 5x5 Kahan matrix with default parameters
%   K = kahan(5);
%   fprintf('Condition number: %.2e\n', cond(K));
%
%   % Well-conditioned version (small theta)
%   K_good = kahan(10, 0.5);
%   fprintf('Condition (theta=0.5): %.2e\n', cond(K_good));
%
%   % Ill-conditioned version (large theta)
%   K_bad = kahan(10, 1.5);
%   fprintf('Condition (theta=1.5): %.2e\n', cond(K_bad));
%
%   % Diagonal matrix (no perturbation)
%   K_diag = kahan(5, 1.2, 0.0);
%   fprintf('Purely diagonal: %d\n', isequal(K_diag, triu(K_diag, 0)));
%
% References:
%   [1] Nicholas J. Higham, "Accuracy and Stability of Numerical
%       Algorithms", 2nd ed., SIAM, 2002, Chapter 28.
%   [2] W. Kahan, Numerical Linear Algebra, Canadian Math. Bulletin,
%       9 (1966), pp. 757-801.
%   [3] NIST Matrix Market: Kahan Matrix,
%       https://math.nist.gov/MatrixMarket/deli/Kahan/information.html
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: TBD
% Version: 0.1.0
% Date: TBD
% Assisted by: Claude Code (Anthropic)
% Compatible with: MATLAB, Octave

% Parse input arguments
if nargin < 1
    error('kahan:TooFewInputs', 'At least one input argument required');
end

% Default for n
if nargin < 2 || isempty(n)
    n = m;
end

% Default parameters
theta = 1.2;
pert = 25;

% Parse optional arguments
if nargin >= 3
    theta = varargin{1};
end
if nargin >= 4
    pert = varargin{2};
end

% Validate inputs
if m < 1 || n < 1
    error('kahan:InvalidInput', 'Dimensions must be positive, got %d x %d', m, n);
end

% Compute sin and cos
s = sin(theta);
c = cos(theta);
r = min(m, n);

% Create rectangular Kahan matrix
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
    K(i, i) = K(i, i) + pert * eps * (r - i + 1);
end

end
