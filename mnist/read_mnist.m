function [trainLabels, trainImages, testLabels, testImages] = read_mnist(dataDir)
%READ_MNIST  Load the original MNIST data files with a sanity-check on the
%           IDX magic numbers.
%
%  Syntax
%  ------
%    [trainL, trainI, testL, testI] = read_mnist()
%    [trainL, trainI, testL, testI] = read_mnist(dataDir)
%
%  Description
%  -----------
%    The function reads the four raw MNIST IDX files that accompany the
%    classic dataset (http://yann.lecun.com/exdb/mnist/):
%
%      * train-labels-idx1-ubyte  - 60000 training labels
%      * train-images-idx3-ubyte  - 60000 training images (28x28 each)
%      * t10k-labels-idx1-ubyte   - 10000 test labels
%      * t10k-images-idx3-ubyte   - 10000 test images  (28x28 each)
%
%  Input arguments
%  ---------------
%    dataDir     - (optional) Directory containing the .ubyte files.
%                  Default: '.' (current directory).
%
%  Output arguments
%  ----------------
%    trainLabels - 60000-by-1 uint8 vector (labels 0-9 unchanged).
%    trainImages - 28-by-28-by-60000 uint8 array (pixel values 0-255).
%    testLabels  - 10000-by-1 uint8 vector (label 0-9 unchanged).
%    testImages  - 28-by-28-by-10000 uint8 array.
%
%  Notes
%  -----
%     The function expects the four .ubyte files to be present in the
%     specified dataDir folder.  It does **not** download or unzip them.
%   * All integer fields in IDX files are big-endian; MATLAB reads them
%     using the 'ieee-be' flag.
%   * If a file's magic number does not match the expected value
%     (2049 for labels, 2051 for images) a warning is issued.
%
%  Example
%  -------
%    [trainL, trainI, testL, testI] = read_mnist();
%    figure(); imshow(trainI(:,:,1)', []);           % show first training image
%    title(['Training label = ' num2str(trainL(1))]);
%    figure(); imshow(testI(:,:,1)', []);            % first test image
%    title(['Test label = ' num2str(testL(1))]);
%
%  See also: fopen, fread, reshape, imshow
%

% ----------------------------------------------------------------------
% Author: <your name>
% Date  : <creation date>
% ----------------------------------------------------------------------

    % Default dataDir to current directory
    if nargin < 1 || isempty(dataDir)
        dataDir = '.';
    end

    %-------------------------------------------------------------
    % 1) TRAIN LABELS   (no conversion)
    %-------------------------------------------------------------
    [trainLabels, magicL] = read_idx_data(fullfile(dataDir, 'train-labels-idx1-ubyte'), ...
                                          60000, []);   % [] -> label file
    fprintf('train-labels  : magic=%d, count=%d\n', magicL, numel(trainLabels));
    if magicL ~= 2049
        warning('Unexpected magic number %d in train-labels file - no label conversion performed.', magicL);
    end
    %-------------------------------------------------------------
    % 2) TRAIN IMAGES
    %-------------------------------------------------------------
    [trainImages, magicI] = read_idx_data(fullfile(dataDir, 'train-images-idx3-ubyte'), ...
                                          60000, [28 28]);
    fprintf('train-images  : magic=%d, count=%d, nrows=%d, ncols=%d\n', ...
            magicI, size(trainImages,3), size(trainImages,1), size(trainImages,2));
    if magicI ~= 2051
        warning('Unexpected magic number %d in train-images file.', magicI);
    end
    %-------------------------------------------------------------
    % 3) TEST LABELS     (no conversion)
    %-------------------------------------------------------------
    [testLabels, magicL] = read_idx_data(fullfile(dataDir, 't10k-labels-idx1-ubyte'), ...
                                         10000, []);
    fprintf('test-labels   : magic=%d, count=%d\n', magicL, numel(testLabels));
    if magicL ~= 2049
        warning('Unexpected magic number %d in test-labels file - label conversion skipped.', magicL);
    end
    %-------------------------------------------------------------
    % 4) TEST IMAGES
    %-------------------------------------------------------------
    [testImages, magicI] = read_idx_data(fullfile(dataDir, 't10k-images-idx3-ubyte'), ...
                                         10000, [28 28]);
    fprintf('test-images   : magic=%d, count=%d, nrows=%d, ncols=%d\n', ...
            magicI, size(testImages,3), size(testImages,1), size(testImages,2));
    if magicI ~= 2051
        warning('Unexpected magic number %d in test-images file.', magicI);
    end
end

%=====================================================================
% Helper: read the header (magic number, count, optional dimensions)
%=====================================================================
function [magic, count, dims] = read_idx_header(fid)
%READ_IDX_HEADER  Extract the IDX header fields from an open file.
%
%  Input
%    fid - file identifier returned by fopen (opened with 'ieee-be').
%
%  Output
%    magic - 32-bit integer (2049 = labels, 2051 = images).
%    count - number of items stored in the file.
%    dims  - [] for label files; [nRows nCols] for image files.
%
%  The file remains open; the caller must close it.
    magic = fread(fid, 1, 'int32');
    count = fread(fid, 1, 'int32');
    dims  = [];
    if magic == 2051               % image file - read two extra dimension fields
        nRows = fread(fid, 1, 'int32');
        nCols = fread(fid, 1, 'int32');
        dims  = [nRows nCols];
    end
end

%=====================================================================
% Helper: read payload and return data together with its magic number
%=====================================================================
function [data, magic] = read_idx_data(fname, expectedCount, imgDims)
%READ_IDX_DATA  Read an IDX file (labels or images) and return its contents.
%
%  Syntax
%    [data, magic] = read_idx_data(fname, expectedCount, imgDims)
%
%  Input
%    fname         - name of the *.ubyte file.
%    expectedCount - number of items the caller expects (e.g. 60000).
%    imgDims       - [] for label files; [nRows nCols] for image files.
%
%  Output
%    data  - column vector of uint8 labels or 3-D uint8 image array.
%    magic - the 32-bit magic number read from the file header.
%
%  The function opens the file in big-endian mode, validates the reported
%  item count (issues a warning if it differs from expectedCount), then
%  reads the raw uint8 payload and reshapes it accordingly.
    fid = fopen(fname, 'r', 'ieee-be');
    if fid == -1
        error('Cannot open file %s', fname);
    end
    % ---- header ----------------------------------------------------
    [magic, count, dimsFromHeader] = read_idx_header(fid);
    if count ~= expectedCount
        warning('File %s reports %d items, but %d were expected.', ...
                fname, count, expectedCount);
    end
    % ---- payload ---------------------------------------------------
    if isempty(imgDims)                     % label file
        raw  = fread(fid, count, 'uint8');
        data = reshape(raw, [], 1);
    else                                    % image file
        if ~isempty(dimsFromHeader)        % sanity: dimensions from header should match caller's
            imgDims = dimsFromHeader;
        end
        elemsPerImg = prod(imgDims);
        raw  = fread(fid, count * elemsPerImg, 'uint8');
        data = reshape(raw, [imgDims, count]);   % rows x cols x count
    end
    fclose(fid);
end
