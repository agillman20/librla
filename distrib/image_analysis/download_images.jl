#=
download_images.jl - Download sample images for image analysis demos.

Description
-----------
Downloads sample images from Pexels (free stock photos) for use with
the image compression demos.

Images are licensed under the Pexels License (free for personal and
commercial use, no attribution required).

Source: https://www.pexels.com/

Requirements
------------
* Packages: Downloads
  using Pkg; Pkg.add("Downloads")

For analysis scripts (test_image_id.jl):
* Packages: Images, FileIO, LinearAlgebra, Plots

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.0.1
Date: April 22, 2026
Assisted by: Claude Code (Anthropic)
=#

using Downloads
using Printf

# Image metadata: (pexels_id, photographer, local_filename)
const IMAGES = [
    (149387, "flickr", "pexels-flickr-149387.jpg"),
    (4793404, "anniroenkae", "pexels-anniroenkae-4793404.jpg"),
    (7824822, "andre-ulysses-de-salis-2100065", "pexels-andre-ulysses-de-salis-2100065-7824822.jpg"),
]

const DATA_DIR = @__DIR__

"""
    get_pexels_url(photo_id)

Construct Pexels download URL for a photo ID.
"""
function get_pexels_url(photo_id::Int)
    return "https://images.pexels.com/photos/$photo_id/pexels-photo-$photo_id.jpeg"
end

"""
    download_image(photo_id, photographer, filename)

Download a single image from Pexels.
"""
function download_image(photo_id::Int, photographer::String, filename::String)
    outpath = joinpath(DATA_DIR, filename)

    if isfile(outpath)
        println("  $filename already exists, skipping")
        return outpath
    end

    url = get_pexels_url(photo_id)
    print("  Downloading $filename...")

    try
        Downloads.download(url, outpath)
        println(" done")
        return outpath
    catch e
        println(" failed ($(typeof(e)))")
        return nothing
    end
end

"""
    download_all_images()

Download all sample images.
"""
function download_all_images()
    println("=" ^ 60)
    println("Image Analysis Sample Data Downloader")
    println("=" ^ 60)
    println("""
This script downloads sample images from Pexels for the
image compression demos.

Images are free to use under the Pexels License.
Source: https://www.pexels.com/
    """)

    downloaded = String[]

    for (photo_id, photographer, filename) in IMAGES
        result = download_image(photo_id, photographer, filename)
        if result !== nothing
            push!(downloaded, result)
        end
    end

    println("\n" * "=" ^ 60)
    println("Download complete: $(length(downloaded))/$(length(IMAGES)) images")
    println("=" ^ 60)

    return downloaded
end

"""
    list_local_images()

List all downloaded images.
"""
function list_local_images()
    if !isdir(DATA_DIR)
        return String[]
    end
    files = filter(f -> startswith(f, "pexels-") && endswith(f, ".jpg"), readdir(DATA_DIR))
    return sort([joinpath(DATA_DIR, f) for f in files])
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    files = download_all_images()

    # Show what we have
    local_files = list_local_images()
    if !isempty(local_files)
        println("\nLocal images: $(length(local_files))")
        for f in local_files
            println("  $(basename(f))")
        end
    end
end
