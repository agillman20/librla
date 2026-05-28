%{
test_sst_modes.m - Modal decomposition of raw NOAA SST data using SVD.

Description
-----------
Applies SVD/EOF analysis to raw Sea Surface Temperature data (no anomaly
preprocessing). The dominant modes capture:

  1. EOF1: Mean spatial pattern (warm tropics, cold poles)
  2. EOF2-4: Seasonal variations
  3. EOF5: ENSO (El Nino-Southern Oscillation)

This demonstrates randomized SVD (svd_sketch) on real climate data.

Data: Raw SST values in C (no anomaly computation, no detrending)

EOF (Empirical Orthogonal Functions) = PCA = POD = SVD

Prerequisites
-------------
Run download_sst_data.m first:
  download_sst_data

Requirements
------------
* MATLAB R2014b+ with NetCDF support (built-in)
* Statistics and Machine Learning Toolbox (for corr function)
* librla.m (automatically added to path)

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.0.1
Date: April 22, 2026
Assisted by: Claude Code (Anthropic)
%}

% Add parent directory to path for librla
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'matlab'));

% ========================================================================
% Load SST data
% ========================================================================

fprintf('%s\n', repmat('=', 1, 70));
fprintf('NOAA SST Climate Mode Analysis (EOF/SVD)\n');
fprintf('%s\n', repmat('=', 1, 70));

DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');

% Find all downloaded files
nc_files = dir(fullfile(DATA_DIR, 'ersst.v5.*.nc'));
nc_files = sort({nc_files.name});

if isempty(nc_files)
    error(['No data files found in ', DATA_DIR, '\n\n', ...
           'Please run download_sst_data.m first:\n', ...
           '  download_sst_data']);
end

fprintf('\nFound %d monthly SST files\n', length(nc_files));
fprintf('  First: %s\n', nc_files{1});
fprintf('  Last:  %s\n', nc_files{end});

% Load first file to get grid info
first_file = fullfile(DATA_DIR, nc_files{1});
lon = ncread(first_file, 'lon');
lat = ncread(first_file, 'lat');

n_lon = length(lon);
n_lat = length(lat);
n_time = length(nc_files);

fprintf('\nGrid dimensions:\n');
fprintf('  Longitude: %d points (%.1f to %.1f)\n', n_lon, lon(1), lon(end));
fprintf('  Latitude:  %d points (%.1f to %.1f)\n', n_lat, lat(1), lat(end));
fprintf('  Time:      %d months\n', n_time);

% Load all SST data
fprintf('\nLoading SST data...\n');
SST = zeros(n_lon, n_lat, n_time, 'single');
dates = NaT(n_time, 1);

for i = 1:n_time
    f = fullfile(DATA_DIR, nc_files{i});
    sst_raw = ncread(f, 'sst');
    SST(:, :, i) = single(sst_raw(:, :, 1, 1));  % lon x lat x lev x time

    % Extract date from filename (ersst.v5.YYYYMM.nc)
    tokens = regexp(nc_files{i}, 'ersst\.v5\.(\d{4})(\d{2})\.nc', 'tokens');
    if ~isempty(tokens)
        dates(i) = datetime(str2double(tokens{1}{1}), str2double(tokens{1}{2}), 1);
    end

    if mod(i, 100) == 0
        fprintf('  Loaded %d / %d\r', i, n_time);
    end
end
fprintf('  Loaded %d months              \n', n_time);

% Handle missing values (land = NaN in ERSST)
SST(SST < -900) = NaN;  % Missing value flag

% Center on Greenwich meridian: convert 0-360 to -180 to 180
fprintf('\nCentering on Greenwich meridian...\n');
shift_idx = find(lon >= 180, 1);
lon = [lon(shift_idx:end) - 360; lon(1:shift_idx-1)];
SST = cat(1, SST(shift_idx:end, :, :), SST(1:shift_idx-1, :, :));

fprintf('  Longitude: %.1f to %.1f\n', lon(1), lon(end));
valid_sst = SST(~isnan(SST));
fprintf('\nSST range: %.1f C to %.1f C\n', min(valid_sst), max(valid_sst));

% ========================================================================
% Create ocean mask and reshape for SVD
% ========================================================================

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('Preparing data matrix...\n');
fprintf('%s\n', repmat('=', 1, 70));

% Ocean mask (where we have valid data for all times)
ocean_mask = ~any(isnan(SST), 3);
n_ocean = sum(ocean_mask(:));

fprintf('\n  Ocean points: %d / %d (%.1f%%)\n', n_ocean, n_lon * n_lat, ...
        100*n_ocean/(n_lon*n_lat));
fprintf('  Area weighting: None (unweighted EOF analysis)\n');

% Reshape to 2D matrix: ocean_points x time
SST_matrix = zeros(n_ocean, n_time, 'single');
ocean_idx = find(ocean_mask);

for t = 1:n_time
    sst_t = SST(:, :, t);
    SST_matrix(:, t) = sst_t(ocean_idx);
end
%%%SST_matrix = detrend(SST_matrix,0);

fprintf('  Data matrix: %d x %d\n', n_ocean, n_time);

% ========================================================================
% EOF Analysis via SVD
% ========================================================================

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('EOF Analysis (SVD)\n');
fprintf('%s\n', repmat('=', 1, 70));

% Number of modes to compute
n_modes = 30;

fprintf('\nComputing %d-mode SVD using svd_sketch...\n', n_modes);
tic;
[U, s, V] = librla.svd_sketch(double(SST_matrix), n_modes, 'power_iter', 2, 'extra_samples', 10);
fprintf('  Elapsed time: %.2fs\n', toc);

% Also compute reference SVD
fprintf('\nComputing reference SVD...\n');
tic;
[U_full, s_full, V_full] = svd(double(SST_matrix), 'econ');
s_full = diag(s_full);
fprintf('  Elapsed time: %.2fs\n', toc);

% Variance explained
total_var = sum(s_full.^2);
var_explained = s_full.^2 / total_var;
cumulative_var = cumsum(var_explained);

fprintf('\nEOF variance explained:\n');
fprintf('  Mode    Variance %%   Cumulative %%\n');
fprintf('  %s\n', repmat('-', 1, 40));
for i = 1:min(10, n_modes)
    fprintf('  EOF%2d   %6.2f%%      %6.2f%%\n', i, 100*var_explained(i), 100*cumulative_var(i));
end

n90 = find(cumulative_var >= 0.90, 1);
n95 = find(cumulative_var >= 0.95, 1);
fprintf('\n  Modes for 90%%: %d, 95%%: %d\n', n90, n95);

% North's rule of thumb for mode separability
test_sst_utils.print_north_test(s_full, n_time, 10);

% ========================================================================
% Reshape EOFs back to spatial maps
% ========================================================================

% Compute Nino 3.4 index for ENSO validation (not for sign fixing)
% Nino 3.4 region: 5N-5S, 170W-120W (-170 to -120 in -181.0.0 system)
nino34_lon_idx = find(lon >= -170 & lon <= -120);
nino34_lat_idx = find(lat >= -5 & lat <= 5);

nino34_index = zeros(n_time, 1);
for t = 1:n_time
    region = SST(nino34_lon_idx, nino34_lat_idx, t);
    nino34_index(t) = mean(region(~isnan(region)));
end

% Fix signs based on spatial pattern mean
% Convention: spatial pattern should have positive mean over ocean
% This ensures positive PC = positive anomaly contribution

% Deterministic SVD
U_det = U_full(:, 1:n_modes);
V_det = V_full(:, 1:n_modes);
for i = 1:n_modes
    if mean(U_det(:, i)) < 0
        U_det(:, i) = -U_det(:, i);
        V_det(:, i) = -V_det(:, i);
    end
end

% Randomized SVD
U_rand = U;
V_rand = V;
for i = 1:n_modes
    if mean(U_rand(:, i)) < 0
        U_rand(:, i) = -U_rand(:, i);
        V_rand(:, i) = -V_rand(:, i);
    end
end

% Spatial: U (non-dimensional pattern)
% Temporal: projection of data onto U, normalized by sqrt(n_ocean) for C units
EOF_maps_det = cell(n_modes, 1);
PC_det = cell(n_modes, 1);
EOF_maps_rand = cell(n_modes, 1);
PC_rand = cell(n_modes, 1);

for i = 1:n_modes
    % Reshape to map
    field_det = NaN(n_lon, n_lat, 'single');
    field_det(ocean_idx) = U_det(:, i);
    EOF_maps_det{i} = field_det;
    PC_det{i} = (U_det(:, i)' * double(SST_matrix))' / sqrt(n_ocean);

    field_rand = NaN(n_lon, n_lat, 'single');
    field_rand(ocean_idx) = U_rand(:, i);
    EOF_maps_rand{i} = field_rand;
    PC_rand{i} = (U_rand(:, i)' * double(SST_matrix))' / sqrt(n_ocean);
end

% ========================================================================
% Visualize Deterministic SVD results
% ========================================================================

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('Visualizing EOF patterns...\n');
fprintf('%s\n', repmat('=', 1, 70));

% Convert dates to decimal years for plotting
years = year(dates) + (month(dates) - 1) / 12;

% Mode labels
mode_names = {'EOF1', 'EOF2', 'EOF3', 'EOF4', 'EOF5'};

figure('Name', 'Deterministic SVD', 'Position', [50, 50, 1400, 1000]);
for i = 1:5
    % Left: Spatial EOF pattern
    subplot(5, 2, 2*i-1);
    clim_val = max(abs(EOF_maps_det{i}(:)), [], 'omitnan') * 0.8;
    pcolor(lon, lat, EOF_maps_det{i}');
    shading flat;
    colormap(gca, redblue());
    caxis([-clim_val, clim_val]);
    xlabel('Longitude');
    ylabel('Latitude');
    title(sprintf('%s (%.1f%%)', mode_names{i}, 100*var_explained(i)));
    axis equal tight;

    % Right: Temporal PC coefficient
    subplot(5, 2, 2*i);
    plot(years, PC_det{i}, 'LineWidth', 0.8, 'Color', [0.27, 0.51, 0.71]);
    xlabel('Year');
    ylabel(sprintf('PC%d', i));
    if i > 1
        yline(0, '--', 'Color', [0.5, 0.5, 0.5]);
    end
    xlim([min(years), max(years)]);
end
sgtitle('Deterministic SVD', 'FontWeight', 'bold');

% ========================================================================
% Visualize Randomized SVD results
% ========================================================================

var_explained_rand = s.^2 / total_var;

figure('Name', 'Randomized SVD', 'Position', [100, 100, 1400, 1000]);
for i = 1:5
    % Left: Spatial EOF pattern
    subplot(5, 2, 2*i-1);
    clim_val = max(abs(EOF_maps_rand{i}(:)), [], 'omitnan') * 0.8;
    pcolor(lon, lat, EOF_maps_rand{i}');
    shading flat;
    colormap(gca, redblue());
    caxis([-clim_val, clim_val]);
    xlabel('Longitude');
    ylabel('Latitude');
    title(sprintf('%s (%.1f%%)', mode_names{i}, 100*var_explained_rand(i)));
    axis equal tight;

    % Right: Temporal PC coefficient
    subplot(5, 2, 2*i);
    plot(years, PC_rand{i}, 'LineWidth', 0.8, 'Color', [0.27, 0.51, 0.71]);
    xlabel('Year');
    ylabel(sprintf('PC%d', i));
    if i > 1
        yline(0, '--', 'Color', [0.5, 0.5, 0.5]);
    end
    xlim([min(years), max(years)]);
end
sgtitle('Randomized SVD (svd\_sketch)', 'FontWeight', 'bold');

% ========================================================================
% Singular value spectrum and variance comparison
% ========================================================================

figure('Name', 'Singular Values', 'Position', [150, 150, 600, 400]);

% Singular value spectrum
semilogy(1:min(50, length(s_full)), s_full(1:min(50, length(s_full))), 'o', ...
         'MarkerSize', 4, 'DisplayName', 'Deterministic');
hold on;
semilogy(1:n_modes, s, 'x', 'MarkerSize', 8, 'Color', 'r', 'DisplayName', 'Randomized');
hold off;
xlabel('Mode');
ylabel('\sigma_i');
title('Singular Value Spectrum');
legend('Location', 'northeast');

% ========================================================================
% ENSO region analysis
% ========================================================================

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('ENSO Analysis\n');
fprintf('%s\n', repmat('=', 1, 70));

% Correlation without detrending
corr_pc5_nino_raw = corr(PC_det{5}, nino34_index);
fprintf('\nCorrelation between PC5 and Nino 3.4 index: %.3f\n', corr_pc5_nino_raw);

% Detrend to remove warming trend and reveal ENSO signal
pc5_detrend = detrend(PC_det{5});
nino34_detrend = detrend(nino34_index);

% Correlation between detrended PC5 (ENSO mode) and Nino 3.4
corr_pc5_nino = corr(pc5_detrend, nino34_detrend);
fprintf('Correlation (detrended): %.3f\n', corr_pc5_nino);

% Plot comparison (detrended and normalized)
figure('Name', 'ENSO Comparison', 'Position', [200, 200, 900, 400]);
plot(years, pc5_detrend / std(pc5_detrend), 'LineWidth', 1.5, 'Color', 'b', 'DisplayName', 'PC5 (det)');
hold on;
plot(years, nino34_detrend / std(nino34_detrend), '--', 'LineWidth', 1.5, 'Color', 'r', 'DisplayName', 'Nino 3.4');
yline(0, ':', 'Color', [0.5, 0.5, 0.5]);
hold off;
xlabel('Year');
ylabel('Index (detrended, normalized)');
title(sprintf('PC5 (ENSO) vs Nino 3.4 Index (r = %.2f, detrended)', corr_pc5_nino));
legend('Location', 'northwest');
xlim([min(years), max(years)]);

% ========================================================================
% Summary
% ========================================================================

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('Summary\n');
fprintf('%s\n', repmat('=', 1, 70));
fprintf('  Data: NOAA ERSST v5\n');
fprintf('  Grid: %d x %d (2 deg resolution)\n', n_lon, n_lat);
fprintf('  Time: %s to %s (%d months)\n', datestr(dates(1), 'yyyy-mm'), ...
        datestr(dates(end), 'yyyy-mm'), n_time);
fprintf('  Ocean points: %d\n', n_ocean);
fprintf('\n');
fprintf('  EOF Analysis:\n');
fprintf('    EOF1 variance: %.1f%% (mean pattern)\n', 100*var_explained(1));
fprintf('    EOF5 variance: %.1f%% (ENSO)\n', 100*var_explained(5));
fprintf('    Modes for 90%% variance: %d\n', n90);
separable = test_sst_utils.north_test(s_full, n_time);
n_well_separated = sum(separable(1:10));
fprintf('    Well-separated modes (North''s rule): %d/10\n', n_well_separated);
fprintf('\n');
fprintf('  Randomized SVD accuracy:\n');
rel_err = norm(s - s_full(1:n_modes)) / norm(s_full(1:n_modes));
fprintf('    Singular value error: %.3f%%\n', 100*rel_err);
fprintf('\n');
fprintf('  ENSO validation (PC5 - Nino 3.4 correlation):\n');
pc5_rand_detrend = detrend(PC_rand{5});
corr_pc5_rand_raw = corr(PC_rand{5}, nino34_index);
corr_pc5_rand = corr(pc5_rand_detrend, nino34_detrend);
fprintf('    Deterministic SVD: %.3f (detrended: %.3f)\n', corr_pc5_nino_raw, corr_pc5_nino);
fprintf('    Randomized SVD:    %.3f (detrended: %.3f)\n', corr_pc5_rand_raw, corr_pc5_rand);
fprintf('%s\n', repmat('=', 1, 70));

% ========================================================================
% Helper function: Red-Blue colormap
% ========================================================================
function cmap = redblue(m)
    if nargin < 1
        m = 256;
    end
    % Create a red-white-blue colormap
    r = [linspace(0, 1, m/2), ones(1, m/2)]';
    g = [linspace(0, 1, m/2), linspace(1, 0, m/2)]';
    b = [ones(1, m/2), linspace(1, 0, m/2)]';
    cmap = [r, g, b];
end
