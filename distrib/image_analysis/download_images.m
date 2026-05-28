%{
download_images.m - Download sample images for image analysis demos.

Description
-----------
Downloads sample images from Pexels (free stock photos) for use with
the image compression demos.

Images are licensed under the Pexels License (free for personal and
commercial use, no attribution required).

Source: https://www.pexels.com/

Requirements
------------
* MATLAB R2014b+ (for websave)
* No additional toolboxes required for downloading

For analysis scripts (test_image_id.m):
* Image Processing Toolbox (for imread) or Octave with image package

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.0.1
Date: April 22, 2026
Assisted by: Claude Code (Anthropic)
%}

function download_images()
    DATA_DIR = fileparts(mfilename('fullpath'));

    % Image metadata: {pexels_id, photographer, local_filename}
    IMAGES = {
        149387, 'flickr', 'pexels-flickr-149387.jpg';
        4793404, 'anniroenkae', 'pexels-anniroenkae-4793404.jpg';
        7824822, 'andre-ulysses-de-salis-2100065', 'pexels-andre-ulysses-de-salis-2100065-7824822.jpg'
    };

    fprintf('%s\n', repmat('=', 1, 60));
    fprintf('Image Analysis Sample Data Downloader\n');
    fprintf('%s\n', repmat('=', 1, 60));
    fprintf('\n');
    fprintf('This script downloads sample images from Pexels for the\n');
    fprintf('image compression demos.\n\n');
    fprintf('Images are free to use under the Pexels License.\n');
    fprintf('Source: https://www.pexels.com/\n\n');

    downloaded = {};

    for i = 1:size(IMAGES, 1)
        photo_id = IMAGES{i, 1};
        photographer = IMAGES{i, 2};
        filename = IMAGES{i, 3};

        result = download_image(photo_id, photographer, filename, DATA_DIR);
        if ~isempty(result)
            downloaded{end+1} = result;
        end
    end

    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('Download complete: %d/%d images\n', length(downloaded), size(IMAGES, 1));
    fprintf('%s\n', repmat('=', 1, 60));

    % Show what we have
    local_files = list_local_images(DATA_DIR);
    if ~isempty(local_files)
        fprintf('\nLocal images: %d\n', length(local_files));
        for i = 1:length(local_files)
            [~, name, ext] = fileparts(local_files{i});
            fprintf('  %s%s\n', name, ext);
        end
    end
end

function url = get_pexels_url(photo_id)
    % Construct Pexels download URL for a photo ID.
%%%    url = sprintf('https://images.pexels.com/photos/%d/pexels-photo-%d.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2', photo_id, photo_id);
    url = sprintf('https://images.pexels.com/photos/%d/pexels-photo-%d.jpeg', photo_id, photo_id);
end

function result = download_image(photo_id, photographer, filename, DATA_DIR)
    % Download a single image from Pexels.
    outpath = fullfile(DATA_DIR, filename);

    if exist(outpath, 'file')
        fprintf('  %s already exists, skipping\n', filename);
        result = outpath;
        return;
    end

    url = get_pexels_url(photo_id);
    fprintf('  Downloading %s...', filename);

    try
        websave(outpath, url);
        fprintf(' done\n');
        result = outpath;
    catch e
        fprintf(' failed (%s)\n', e.message);
        result = '';
    end
end

function files = list_local_images(DATA_DIR)
    % List all downloaded images.
    if nargin < 1
        DATA_DIR = fileparts(mfilename('fullpath'));
    end

    if ~exist(DATA_DIR, 'dir')
        files = {};
        return;
    end

    listing = dir(fullfile(DATA_DIR, 'pexels-*.jpg'));
    files = sort({listing.name});
    files = cellfun(@(f) fullfile(DATA_DIR, f), files, 'UniformOutput', false);
end
