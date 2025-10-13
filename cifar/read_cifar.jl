# ------------------------------------------------------------
# read_cifar.jl
# ------------------------------------------------------------
# 2025‑10‑12
#
#   Load the CIFAR‑10 dataset from MATLAB .mat files and return
#   them as Julia arrays.
#
#   CIFAR‑10 consists of 60000 32×32 color images in 10 classes,
#   with 6000 images per class. There are 50000 training images
#   and 10000 test images.
#
#   Dataset source: https://www.cs.toronto.edu/~kriz/cifar.html
#
#   • No download or unzip step is performed – the files must already
#     exist in the directory given by `dir`.
#   • Labels are **not** changed (values 0‑9).
#   • Images are returned twice:
#         – as raw `UInt8` tensors (`train_images`, `test_images`);
#         – as `Float32` tensors scaled to `[0,1]` (`train_X`, `test_X`).
#
using MAT              # for reading .mat files
using Printf           # nice formatted output

# ---------- Main loader ----------
function read_cifar(
        dir::AbstractString = ".";        # directory where cifar data sits
        data_subdir::AbstractString = "cifar-10-batches-mat"
    )

    data_path = joinpath(dir, data_subdir)

    # Check if directory exists
    if !isdir(data_path)
        error("Directory $data_path not found. Please ensure CIFAR-10 data is downloaded.")
    end

    # ---- Load metadata (class names) ------------------------
    meta_file = joinpath(data_path, "batches.meta.mat")
    if !isfile(meta_file)
        error("Metadata file $meta_file not found.")
    end

    meta = matread(meta_file)
    class_names = meta["label_names"]
    @printf "Loaded metadata: %d classes\n" length(class_names)

    # ---- Load training batches (5 batches × 10000 images each) ----
    n_train_batches = 5
    train_labels = UInt8[]
    train_data = Array{UInt8}(undef, 0, 3072)

    for i_batch in 1:n_train_batches
        batch_file = joinpath(data_path, "data_batch_$(i_batch).mat")
        if !isfile(batch_file)
            error("Training batch file $batch_file not found.")
        end

        batch = matread(batch_file)

        # Append labels and data
        # MATLAB stores labels as column vector, convert to UInt8
        batch_labels = UInt8.(vec(batch["labels"]))
        append!(train_labels, batch_labels)

        # Append data (should be 10000×3072)
        batch_data = batch["data"]
        train_data = vcat(train_data, batch_data)

        batch_label = get(batch, "batch_label", "batch $i_batch")
        @printf "Loaded %s: %d images\n" batch_label size(batch_data, 1)
    end

    # Reshape training data from (N×3072) to (32×32×3×N)
    train_images = reshape_cifar_images(train_data)
    @printf "Training set: %d images total\n" size(train_images, 4)

    # ---- Load test batch ------------------------
    test_file = joinpath(data_path, "test_batch.mat")
    if !isfile(test_file)
        error("Test batch file $test_file not found.")
    end

    test_batch = matread(test_file)
    test_labels = UInt8.(vec(test_batch["labels"]))

    # Reshape test data from (N×3072) to (32×32×3×N)
    test_images = reshape_cifar_images(test_batch["data"])
    @printf "Test set: %d images\n" size(test_images, 4)

    # ---- Summary ------------------------
    @printf "\nCIFAR-10 dataset loaded successfully:\n"
    @printf "  Training: %d images, %d labels\n" size(train_images, 4) length(train_labels)
    @printf "  Test:     %d images, %d labels\n" size(test_images, 4) length(test_labels)
    @printf "  Classes:  %d (%s, ..., %s)\n" length(class_names) class_names[1] class_names[end]

    # ------------------------------------------------------------
    # Optional: convert to Float32 in [0, 1] for most ML frameworks
    # ------------------------------------------------------------
    train_X = Float32.(train_images) ./ 255
    test_X  = Float32.(test_images)  ./ 255

    # ------------------------------------------------------------
    # Return everything (keep the original UInt8 version if you wish)
    # ------------------------------------------------------------
    return (
        train_labels, train_X,
        test_labels , test_X,
        class_names
    )
end

# ------------------------------------------------------------
# Helper: Reshape flattened CIFAR data to image arrays
# ------------------------------------------------------------
function reshape_cifar_images(data::AbstractMatrix{T}) where T
    """
    Convert (N×3072) array to (32×32×3×N).

    Input
    -----
    data : N×3072 matrix (typically UInt8)
           Each row is a flattened 32×32×3 image.
           The first 1024 values are the red channel (row-major),
           next 1024 are green, last 1024 are blue.

    Output
    ------
    images : 32×32×3×N array (same type as input)
             images[:,:,:,i] is the i-th RGB image.

    The CIFAR-10 format stores each 32×32×3 image as a 3072-element
    row vector: [R0...R1023, G0...G1023, B0...B1023] where each channel
    is stored in row-major order (C-style).
    """

    N = size(data, 1)

    # Extract RGB channels (each N×1024)
    R = data[:, 1:1024]
    G = data[:, 1025:2048]
    B = data[:, 2049:3072]

    # CIFAR stores in row-major order (C-style), but Julia uses column-major
    # Reshape each channel from (N, 1024) to (32, 32, N)
    # Since CIFAR is row-major, we transpose to get correct layout
    R_img = reshape(R', 32, 32, N)
    G_img = reshape(G', 32, 32, N)
    B_img = reshape(B', 32, 32, N)

    # Allocate output array (32×32×3×N)
    images = Array{T}(undef, 32, 32, 3, N)

    # Fill in RGB channels and transpose spatial dimensions for correct orientation
    for i in 1:N
        images[:, :, 1, i] = transpose(R_img[:, :, i])
        images[:, :, 2, i] = transpose(G_img[:, :, i])
        images[:, :, 3, i] = transpose(B_img[:, :, i])
    end

    return images
end

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    train_lbl, train_X, test_lbl, test_X, names = read_cifar(".")

    @printf "\nCIFAR-10 data loaded successfully.\n"
    @printf "train labels: %d entries (first 10: %s)\n" length(train_lbl) string(train_lbl[1:10])
    @printf "train images: %d×%d×%d×%d (%s)\n" size(train_X)... string(eltype(train_X))
    @printf "test  labels: %d entries (first 10: %s)\n" length(test_lbl) string(test_lbl[1:10])
    @printf "test  images: %d×%d×%d×%d (%s)\n" size(test_X)... string(eltype(test_X))
    @printf "\nClass names: %s\n" string(names)
end
