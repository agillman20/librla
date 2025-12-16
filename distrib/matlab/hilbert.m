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
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: TBD
% Version: 0.1.0
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
