function exit_code = test_all()
% % add next level up directory to search path
addpath(genpath('..'));
%
%VALIDATE_ALL Master test script for librla
%
%   Runs all test tests and produces a unified summary:
%   - test_id.m    (ID implementations)
%   - test_svd.m   (SVD implementations)
%   - test_qr.m    (QR implementations)
%   - test_orth.m  (Orth implementations)
%
%   Usage:
%       test_all
%       exit_code = test_all()
%
%   Returns 0 if all tests pass, 1 otherwise.
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: NIST-PD
% Version: 1.0.0
% Date: January 5, 2026
% Assisted by: Claude Code (Anthropic)

fprintf('\n');
fprintf('%s\n', repmat('=', 1, 70));
fprintf('LIBRLA TEST SUITE - MATLAB/Octave\n');
fprintf('%s\n', repmat('=', 1, 70));
fprintf('\n');
fprintf('This script runs all test tests for librla functions:\n');
fprintf('  - test_id.m    (id_sketch, id_qrpiv)\n');
fprintf('  - test_svd.m   (svd_sketch)\n');
fprintf('  - test_qr.m    (qr_sketch)\n');
fprintf('  - test_orth.m  (orth_sketch)\n');
fprintf('\n');

% Detect environment
if exist('OCTAVE_VERSION', 'builtin')
    fprintf('Environment: Octave %s\n', OCTAVE_VERSION);
else
    fprintf('Environment: MATLAB %s\n', version);
end
fprintf('%s\n', repmat('=', 1, 70));

% List of test modules
modules = {
    'test_id',   'test_id';
    'test_svd',  'test_svd';
    'test_qr',   'test_qr';
    'test_orth', 'test_orth'
    };

num_modules = size(modules, 1);
results = cell(num_modules, 4);  % {name, status, elapsed, success}
total_elapsed = 0.0;

for i = 1:num_modules
    func_name = modules{i, 1};
    display_name = modules{i, 2};

    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('Running %s...\n', display_name);
    fprintf('%s\n', repmat('=', 1, 70));

    try
        t0 = tic;
        % Call the test function
        module_exit_code = feval(func_name);
        elapsed = toc(t0);
        total_elapsed = total_elapsed + elapsed;

        if module_exit_code == 0
            status = 'PASSED';
            success = true;
        else
            status = 'FAILED';
            success = false;
        end
    catch ME
        elapsed = 0.0;
        status = 'ERROR';
        success = false;
        fprintf('\n[ERROR] Failed to run %s: %s\n', display_name, ME.message);
    end

    results{i, 1} = display_name;
    results{i, 2} = status;
    results{i, 3} = elapsed;
    results{i, 4} = success;
end
% =========================================================================
% FINAL SUMMARY
% =========================================================================
fprintf('\n\n');
fprintf('%s\n', repmat('=', 1, 70));
fprintf('FINAL SUMMARY\n');
fprintf('%s\n', repmat('=', 1, 70));
fprintf('\n');

% Per-module summary
fprintf('%-20s %-12s %-12s\n', 'Module', 'Status', 'Time (s)');
fprintf('%s\n', repmat('-', 1, 50));

all_passed = true;
modules_passed = 0;

for i = 1:num_modules
    display_name = results{i, 1};
    status = results{i, 2};
    elapsed = results{i, 3};

    fprintf('%-20s [%-6s]    %8.2fs\n', display_name, status, elapsed);

    if ~strcmp(status, 'PASSED')
        all_passed = false;
    else
        modules_passed = modules_passed + 1;
    end
end

fprintf('%s\n', repmat('-', 1, 50));
fprintf('%-20s %12s %8.2fs\n', 'Total', '', total_elapsed);

% Overall status
fprintf('\n');
fprintf('%s\n', repmat('=', 1, 70));

if all_passed
    fprintf('[PASS] All %d test modules passed!\n', num_modules);
    fprintf('%s\n', repmat('=', 1, 70));
    exit_code = 0;
else
    fprintf('[FAIL] %d/%d test modules passed\n', modules_passed, num_modules);
    fprintf('\n');
    fprintf('Failed modules:\n');
    for i = 1:num_modules
        if ~strcmp(results{i, 2}, 'PASSED')
            fprintf('  - %s: %s\n', results{i, 1}, results{i, 2});
        end
    end
    fprintf('%s\n', repmat('=', 1, 70));
    exit_code = 1;
end

end
