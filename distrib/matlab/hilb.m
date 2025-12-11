function a = hilb(m,n)
% HILB - Generate Hilbert matrix
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: TBD
% Version: 1.0.0
% Date: TBD
% Assisted by: Claude Code (Anthropic)

  if( nargin == 1 ) n = m; end
  a = zeros(m,n);
  i = [1:n];
  j = [1:m]';
  a = 1./(bsxfun(@plus,i,j)-1);

%%  for j = 1:m
%%    for i = 1:n
%%	a(j,i) = 1/(i+j-1);
%%    end
%%  end

