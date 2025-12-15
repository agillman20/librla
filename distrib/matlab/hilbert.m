function H = hilbert(m, n)
% HILBERT  Generate Hilbert matrix
%
%   H = hilbert(m)
%   H = hilbert(m, n)
%
% The Hilbert matrix is severely ill-conditioned, with entries
% H(i,j) = 1/(i+j-1). Useful for testing numerical stability.
%
% Arguments:
%   m - Number of rows
%   n - Number of columns (default: m)
%
% Returns:
%   H - Hilbert matrix (m x n)
%
% Notes:
%   The condition number grows exponentially with matrix size:
%   cond(H) ~ O((1+sqrt(2))^(4n) / sqrt(n))
%
%   The singular values decay rapidly, making this matrix ideal
%   for testing rank-revealing and low-rank approximation algorithms.
%
% Examples:
%   % 5x5 Hilbert matrix
%   H = hilbert(5);
%   fprintf('Condition number: %.2e\n', cond(H));
%
%   % Rectangular matrix
%   H_rect = hilbert(100, 50);
%   fprintf('Shape: %d x %d\n', size(H_rect));
%
% References:
%   [1] Nicholas J. Higham, "Accuracy and Stability of Numerical
%       Algorithms", 2nd ed., SIAM, 2002, Chapter 28.
%   [2] D. Hilbert, "Ein Beitrag zur Theorie des Legendre'schen
%       Polynoms", Acta Mathematica, 18 (1894), pp. 155-159.
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: TBD
% Version: 1.0.0
% Date: TBD
% Assisted by: Claude Code (Anthropic)
% Compatible with: MATLAB, Octave

% Default for n
if nargin < 2 || isempty(n)
    n = m;
end

% Validate inputs
if m < 1 || n < 1
    error('hilbert:InvalidInput', 'Dimensions must be positive, got %d x %d', m, n);
end

% Generate Hilbert matrix using vectorized computation
i = (1:n);
j = (1:m)';
H = 1.0 ./ (bsxfun(@plus, i, j) - 1);

end
