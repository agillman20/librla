%{
download_sst_data.m - Download NOAA ERSST v5 Sea Surface Temperature data.

Description
-----------
Downloads monthly SST data from NOAA's Extended Reconstructed Sea Surface
Temperature (ERSST) Version 5 dataset.

Data source: https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/
Product info: https://www.ncei.noaa.gov/products/extended-reconstructed-sst

ERSST v5 specifications:
  - Grid: 2 x 2 degrees (89 lat x 180 lon)
  - Coverage: 1854-present
  - Variables: SST (sea surface temperature)
  - Format: NetCDF (CF conventions)

Citation:
  Huang et al. (2017): NOAA Extended Reconstructed Sea Surface Temperature
  (ERSST), Version 5. doi:10.7289/V5T72FNM

Requirements
------------
* MATLAB R2014b+ (for websave)
* No additional toolboxes required for downloading

For analysis scripts (test_sst_*.m):
* Statistics and Machine Learning Toolbox (for corr function)
* Or use: corrcoef(x,y) and extract r=c(1,2) instead

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.0.1
Date: April 22, 2026
Assisted by: Claude Code (Anthropic)
%}

function download_sst_data()
    BASE_URL = 'https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/';
    DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');

    fprintf('%s\n', repmat('=', 1, 60));
    fprintf('NOAA ERSST v5 Data Downloader\n');
    fprintf('%s\n', repmat('=', 1, 60));
    fprintf('\n');
    fprintf('This script downloads monthly Sea Surface Temperature data.\n\n');
    fprintf('Grid: 2 x 2 degrees (89 lat x 180 lon = 16,020 ocean points)\n');
    fprintf('Time: Monthly, 1854-present (~2000 months)\n\n');
    fprintf('Each file is ~200KB, full dataset ~400MB.\n\n');

    % Download recent decades for demo (1980-2023)
    fprintf('Downloading 1980-2023 (44 years, ~530 files)...\n');
    fprintf('This will take a few minutes...\n\n');

    files = download_ersst_range(1980, 2023, BASE_URL, DATA_DIR);

    fprintf('\nTo download more years, run:\n');
    fprintf('  download_ersst_range(1854, 1979)\n');

    % Show what we have
    local_files = list_local_files(DATA_DIR);
    if ~isempty(local_files)
        fprintf('\nLocal files: %d\n', length(local_files));
        [~, first_name, ext] = fileparts(local_files{1});
        [~, last_name, ext2] = fileparts(local_files{end});
        fprintf('  First: %s%s\n', first_name, ext);
        fprintf('  Last:  %s%s\n', last_name, ext2);
    end
end

function files_downloaded = download_ersst_year(year, BASE_URL, DATA_DIR)
    % Download all monthly ERSST files for a given year.
    % Files are named: ersst.v5.YYYYMM.nc

    if ~exist(DATA_DIR, 'dir')
        mkdir(DATA_DIR);
    end

    files_downloaded = {};

    for month = 1:12
        filename = sprintf('ersst.v5.%04d%02d.nc', year, month);
        url = [BASE_URL, filename];
        outpath = fullfile(DATA_DIR, filename);

        if exist(outpath, 'file')
            fprintf('  %s already exists, skipping\n', filename);
            files_downloaded{end+1} = outpath;
            continue;
        end

        fprintf('  Downloading %s...', filename);
        try
            websave(outpath, url);
            fprintf(' done\n');
            files_downloaded{end+1} = outpath;
        catch e
            fprintf(' failed (%s)\n', class(e));
            % File might not exist yet for recent months
        end
    end
end

function all_files = download_ersst_range(start_year, end_year, BASE_URL, DATA_DIR)
    % Download ERSST data for a range of years.

    if nargin < 3
        BASE_URL = 'https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/';
    end
    if nargin < 4
        DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');
    end

    fprintf('%s\n', repmat('=', 1, 60));
    fprintf('Downloading NOAA ERSST v5 data\n');
    fprintf('  Years: %d - %d\n', start_year, end_year);
    fprintf('  Source: %s\n', BASE_URL);
    fprintf('%s\n', repmat('=', 1, 60));

    all_files = {};

    for year = start_year:end_year
        fprintf('\nYear %d:\n', year);
        files = download_ersst_year(year, BASE_URL, DATA_DIR);
        all_files = [all_files, files];
    end

    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('Download complete: %d files\n', length(all_files));
    fprintf('%s\n', repmat('=', 1, 60));
end

function files = list_local_files(DATA_DIR)
    % List all downloaded ERSST files.

    if nargin < 1
        DATA_DIR = fullfile(fileparts(mfilename('fullpath')), 'data');
    end

    if ~exist(DATA_DIR, 'dir')
        files = {};
        return;
    end

    listing = dir(fullfile(DATA_DIR, '*.nc'));
    files = sort({listing.name});
    files = cellfun(@(f) fullfile(DATA_DIR, f), files, 'UniformOutput', false);
end
