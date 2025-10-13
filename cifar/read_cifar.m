function [trainLabels, trainImages, testLabels, testImages, classNames] = read_cifar()
%READ_CIFAR  Load the CIFAR-10 dataset from MATLAB .mat files.
%
%  Syntax
%  ------
%    [trainL, trainI, testL, testI, classNames] = read_cifar()
%
%  Description
%  -----------
%    The function reads the CIFAR-10 dataset files in MATLAB format from
%    the cifar-10-batches-mat/ directory. CIFAR-10 consists of 60000
%    32x32 color images in 10 classes, with 6000 images per class.
%    There are 50000 training images and 10000 test images.
%
%    Dataset source: https://www.cs.toronto.edu/~kriz/cifar.html
%
%    The function loads:
%      * data_batch_1.mat through data_batch_5.mat - 5 training batches
%        (10000 images each = 50000 total)
%      * test_batch.mat - 10000 test images
%      * batches.meta.mat - class names
%
%  Output arguments
%  ----------------
%    trainLabels - 50000-by-1 uint8 vector (labels 0-9 unchanged).
%    trainImages - 32-by-32-by-3-by-50000 uint8 array.
%                  RGB images with dimensions [height, width, channels, count].
%    testLabels  - 10000-by-1 uint8 vector (labels 0-9 unchanged).
%    testImages  - 32-by-32-by-3-by-10000 uint8 array.
%    classNames  - 10-by-1 cell array of strings with class names:
%                  {'airplane', 'automobile', 'bird', 'cat', 'deer',
%                   'dog', 'frog', 'horse', 'ship', 'truck'}
%
%  Notes
%  -----
%   * The function expects the cifar-10-batches-mat/ directory to be
%     present in the current folder.
%   * Each image is 32×32 pixels with 3 color channels (RGB), stored as
%     a flattened 3072-element vector in the .mat files. The first 1024
%     values are red channel, next 1024 are green, last 1024 are blue.
%   * Labels range from 0-9 (no conversion performed).
%   * All data is returned as uint8 (0-255 range).
%
%  Example
%  -------
%  %{
%  [trainL, trainI, testL, testI, names] = read_cifar();
%  % Show first training image
%  figure(); imshow(trainI(:,:,:,1));
%  title(['Training label = ' num2str(trainL(1)) ' (' names{trainL(1)+1} ')']);
%  % Show first test image
%  figure(); imshow(testI(:,:,:,1));
%  title(['Test label = ' num2str(testL(1)) ' (' names{testL(1)+1} ')']);
%  %}
%
%  See also: load, reshape, imshow
%
% ----------------------------------------------------------------------
% Author: Claude Code
% Date  : 2025-10-12
% ----------------------------------------------------------------------

    dataDir = 'cifar-10-batches-mat';

    % Check if directory exists
    if ~exist(dataDir, 'dir')
        error('Directory %s not found. Please ensure CIFAR-10 data is downloaded.', dataDir);
    end

    %-------------------------------------------------------------
    % 1) Load metadata (class names)
    %-------------------------------------------------------------
    metaFile = fullfile(dataDir, 'batches.meta.mat');
    if ~exist(metaFile, 'file')
        error('Metadata file %s not found.', metaFile);
    end
    meta = load(metaFile);
    classNames = meta.label_names;
    fprintf('Loaded metadata: %d classes\n', numel(classNames));

    %-------------------------------------------------------------
    % 2) Load training batches (5 batches × 10000 images each)
    %-------------------------------------------------------------
    nTrainBatches = 5;
    trainLabels = [];
    trainData = [];

    for iBatch = 1:nTrainBatches
        batchFile = fullfile(dataDir, sprintf('data_batch_%d.mat', iBatch));
        if ~exist(batchFile, 'file')
            error('Training batch file %s not found.', batchFile);
        end
        batch = load(batchFile);

        % Append labels and data
        trainLabels = [trainLabels; uint8(batch.labels)];
        trainData = [trainData; batch.data];

        fprintf('Loaded %s: %d images\n', batch.batch_label, size(batch.data, 1));
    end

    % Reshape training data from (N×3072) to (32×32×3×N)
    trainImages = reshape_cifar_images(trainData);
    fprintf('Training set: %d images total\n', size(trainImages, 4));

    %-------------------------------------------------------------
    % 3) Load test batch
    %-------------------------------------------------------------
    testFile = fullfile(dataDir, 'test_batch.mat');
    if ~exist(testFile, 'file')
        error('Test batch file %s not found.', testFile);
    end
    testBatch = load(testFile);
    testLabels = uint8(testBatch.labels);

    % Reshape test data from (N×3072) to (32×32×3×N)
    testImages = reshape_cifar_images(testBatch.data);
    fprintf('Test set: %d images\n', size(testImages, 4));

    %-------------------------------------------------------------
    % 4) Summary
    %-------------------------------------------------------------
    fprintf('\nCIFAR-10 dataset loaded successfully:\n');
    fprintf('  Training: %d images, %d labels\n', size(trainImages, 4), numel(trainLabels));
    fprintf('  Test:     %d images, %d labels\n', size(testImages, 4), numel(testLabels));
    fprintf('  Classes:  %d (%s, ..., %s)\n', numel(classNames), classNames{1}, classNames{end});
end

%=====================================================================
% Helper: Reshape flattened CIFAR data to image arrays
%=====================================================================
function images = reshape_cifar_images(data)
%RESHAPE_CIFAR_IMAGES  Convert (N×3072) uint8 array to (32×32×3×N).
%
%  Input
%    data - N-by-3072 uint8 matrix, where each row is a flattened image.
%           The first 1024 values are the red channel (row-major),
%           next 1024 are green, last 1024 are blue.
%
%  Output
%    images - 32-by-32-by-3-by-N uint8 array.
%             images(:,:,:,i) is the i-th RGB image.
%
%  The CIFAR-10 format stores each 32×32×3 image as a 3072-element
%  row vector: [R0...R1023, G0...G1023, B0...B1023] where each channel
%  is stored in row-major order.

    N = size(data, 1);

    % Extract RGB channels (each N×1024)
    R = data(:, 1:1024);        % Red channel
    G = data(:, 1025:2048);     % Green channel
    B = data(:, 2049:3072);     % Blue channel

    % CIFAR stores in row-major order (C-style), but MATLAB uses column-major
    % Reshape each channel: transpose first to get (1024×N), then reshape to (32×32×N)
    R_img = reshape(R', [32, 32, N]);
    G_img = reshape(G', [32, 32, N]);
    B_img = reshape(B', [32, 32, N]);

    % Allocate output array (32×32×3×N)
    images = zeros(32, 32, 3, N, 'uint8');

    % Fill in RGB channels and transpose spatial dimensions
    for i = 1:N
        images(:,:,1,i) = R_img(:,:,i)';  % Transpose for correct orientation
        images(:,:,2,i) = G_img(:,:,i)';
        images(:,:,3,i) = B_img(:,:,i)';
    end
end
