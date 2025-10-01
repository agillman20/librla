# ------------------------------------------------------------
# read_mnist.jl
# ------------------------------------------------------------
# 2025‑09‑30
#
#   Load the four original MNIST IDX files (train‑labels‑idx1‑ubyte,
#   train‑images‑idx3‑ubyte, t10k‑labels‑idx1‑ubyte, t10k‑images‑idx3‑ubyte)
#   and return them as Julia arrays.
#
#   • No download or unzip step is performed – the files must already
#     exist in the directory given by `dir`.
#   • All integer fields are stored big‑endian; we use `ntoh` to convert
#     them on any platform.
#   • Labels are **not** changed
#   • Images are returned twice:
#         – as raw `UInt8` tensors (`train_images`, `test_images`);
#         – as `Float32` tensors scaled to `[0,1]` (`train_X`, `test_X`).
#
using Sockets          # for ntoh
using Mmap             # optional: memory‑map the files
using Printf           # nice formatted output

# ---------- Helper to read a 32‑bit big‑endian integer ----------
read_be_int32(io) = ntoh(read(io, Int32))

# ---------- Main loader (uses do‑blocks for safe closing) ----------
function read_mnist(
        dir::AbstractString = ".";   # directory where the four .ubyte files sit
        mapfiles::Bool = false       # set true to memory‑map instead of read!
    )

    # ---- Local helper to read label files -----------------------
    function read_labels(fname)
        path = joinpath(dir, fname)
        open(path, "r") do f
            @assert read_be_int32(f) == 2049 "Unexpected magic number in $path"
            n = read_be_int32(f)                 # number of labels
            lbl = Vector{UInt8}(undef, n)
            read!(f, lbl)                        # read raw bytes
            return lbl
        end
    end

    # ---- Load the training / test label vectors ------------------
    train_labels = read_labels("train-labels-idx1-ubyte")
    test_labels  = read_labels("t10k-labels-idx1-ubyte")

    # ---- Local helper to read image files ------------------------
    function read_images(fname)
        path = joinpath(dir, fname)
        open(path, "r") do f
            @assert read_be_int32(f) == 2051 "Unexpected magic number in $path"
            cnt   = read_be_int32(f)   # number of images
            nrow  = read_be_int32(f)   # rows per image
            ncol  = read_be_int32(f)   # columns per image

            # allocate (rows, cols, count) in column‑major order
            img = Array{UInt8}(undef, nrow, ncol, cnt)

            if mapfiles
                # Memory‑map the raw data section only (skip header)
                # Offset = 16 bytes (4 × Int32 header)
                data = Mmap.mmap(path, Array{UInt8},
                                 (nrow * ncol * cnt,);
                                 offset = 16, shared = false)
                reshape!(img, size(img)) .= reshape(data, size(img))
            else
                read!(f, img)
            end
            return img
        end
    end

    # ---- Load the training / test image tensors ------------------
    train_images = read_images("train-images-idx3-ubyte")
    test_images  = read_images("t10k-images-idx3-ubyte")

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
        test_labels , test_X
    )
end

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    train_lbl, train_X, test_lbl, test_X = read_mnist(".")
    @printf "Training set: %d images, %d×%d each, %d labels\n" size(train_X,3) size(train_X,1) size(train_X,2) length(train_lbl)
    @printf "Test set: %d images, %d×%d each, %d labels\n" size(test_X,3) size(test_X,1) size(test_X,2) length(test_lbl)
end
