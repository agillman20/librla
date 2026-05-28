classdef test_sst_utils
% TEST_SST_UTILS - Statistical utilities for SST/EOF climate analysis
%
% Functions for EOF analysis validation:
%   test_sst_utils.north_test       - Apply North's rule of thumb for mode separability
%   test_sst_utils.print_north_test - Print North's rule results
%
% USAGE:
%   separable = test_sst_utils.north_test(s, n_samples);
%   [separable, errors] = test_sst_utils.north_test(s, n_samples);
%   test_sst_utils.print_north_test(s, n_samples);
%   test_sst_utils.print_north_test(s, n_samples, n_modes);
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: MIT
% Version: 1.0.1
% Date: April 22, 2026
% Assisted by: Claude Code (Anthropic)

methods (Static)

function [separable, errors] = north_test(singular_values, n_samples)
% NORTH_TEST - Apply North's rule of thumb to test EOF/SVD mode separability
%
% Syntax:
%   separable = test_sst_utils.north_test(singular_values, n_samples)
%   [separable, errors] = test_sst_utils.north_test(singular_values, n_samples)
%
% Description:
%   North et al. (1982) showed that eigenvalues have sampling error:
%       delta_lambda_i = lambda_i * sqrt(2/N)
%
%   Two modes are considered "effectively degenerate" if their eigenvalues
%   overlap within error bars:
%       lambda_i - lambda_{i+1} < delta_lambda_i + delta_lambda_{i+1}
%
%   For SVD, eigenvalues lambda = sigma^2 (squared singular values).
%
% Input Arguments:
%   singular_values - Singular values from SVD (sigma, not sigma^2)
%   n_samples       - Number of independent samples (typically time steps)
%
% Output Arguments:
%   separable - Logical array; separable(i) is true if mode i is
%               well-separated from mode i+1
%   errors    - Sampling errors delta_lambda for each eigenvalue
%
% Reference:
%   North, G. R., T. L. Bell, R. F. Cahalan, and F. J. Moeng, 1982:
%   Sampling errors in the estimation of empirical orthogonal functions.
%   Mon. Wea. Rev., 110, 699-706.

  s = singular_values(:);
  eigenvalues = s.^2;

  % Sampling error: delta_lambda_i = lambda_i * sqrt(2/N)
  errors = eigenvalues * sqrt(2.0 / n_samples);

  % Test separation: lambda_i - lambda_{i+1} > delta_lambda_i + delta_lambda_{i+1}
  gap = eigenvalues(1:end-1) - eigenvalues(2:end);
  error_sum = errors(1:end-1) + errors(2:end);
  separable = gap > error_sum;
end

function print_north_test(singular_values, n_samples, n_modes)
% PRINT_NORTH_TEST - Print North's rule of thumb results for EOF analysis
%
% Syntax:
%   test_sst_utils.print_north_test(singular_values, n_samples)
%   test_sst_utils.print_north_test(singular_values, n_samples, n_modes)
%
% Description:
%   Prints a formatted table showing variance explained, eigenvalue error
%   bars, and whether each mode pair is well-separated according to North's
%   rule of thumb.
%
% Input Arguments:
%   singular_values - Singular values from SVD
%   n_samples       - Number of independent samples
%   n_modes         - Number of modes to display (default: 10)

  if nargin < 3, n_modes = 10; end

  s = singular_values(:);
  n_modes = min(n_modes, length(s) - 1);

  [separable, errors] = test_sst_utils.north_test(s, n_samples);
  eigenvalues = s.^2;
  total_var = sum(s.^2);
  var_pct = 100 * eigenvalues / total_var;

  fprintf('\nNorth''s Rule of Thumb (N=%d samples)\n', n_samples);
  fprintf('  Mode   Variance %%    lambda +/- delta_lambda     Separated?\n');
  fprintf('  %s\n', repmat('-', 1, 55));

  for i = 1:n_modes
      if separable(i)
          status = 'Yes';
      else
          status = 'NO (degenerate)';
      end
      fprintf('  EOF%2d   %6.2f%%    %.2e +/- %.2e   %s\n', ...
          i, var_pct(i), eigenvalues(i), errors(i), status);
  end

  n_degen = sum(~separable(1:n_modes));
  if n_degen > 0
      fprintf('\n  Warning: %d mode pair(s) may be degenerate\n', n_degen);
  end
end

end

end
