function all_files = download_ersst_range(start_year, end_year, BASE_URL, DATA_DIR)
    % Download ERSST data for a range of years.
    %
    % Usage:
    %   download_ersst_range(1970, 1979)
    %   download_ersst_range(1854, 1979)
    %
    % Author: Adrianna Gillman, Zydrunas Gimbutas
    % SPDX-License-Identifier: MIT
    % Version: 1.0.1
    % Date: April 22, 2026
    % Assisted by: Claude Code (Anthropic)

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

    if ~exist(DATA_DIR, 'dir')
        mkdir(DATA_DIR);
    end

    all_files = {};

    for year = start_year:end_year
        fprintf('\nYear %d:\n', year);
        for month = 1:12
            filename = sprintf('ersst.v5.%04d%02d.nc', year, month);
            url = [BASE_URL, filename];
            outpath = fullfile(DATA_DIR, filename);

            if exist(outpath, 'file')
                fprintf('  %s already exists, skipping\n', filename);
                all_files{end+1} = outpath;
                continue;
            end

            fprintf('  Downloading %s...', filename);
            try
                websave(outpath, url);
                fprintf(' done\n');
                all_files{end+1} = outpath;
            catch e
                fprintf(' failed (%s)\n', class(e));
            end
        end
    end

    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('Download complete: %d files\n', length(all_files));
    fprintf('%s\n', repmat('=', 1, 60));
end
