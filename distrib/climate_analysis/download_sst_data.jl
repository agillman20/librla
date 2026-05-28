#=
download_sst_data.jl - Download NOAA ERSST v5 Sea Surface Temperature data.

Description
-----------
Downloads monthly SST data from NOAA's Extended Reconstructed Sea Surface
Temperature (ERSST) Version 5 dataset.

Data source: https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/
Product info: https://www.ncei.noaa.gov/products/extended-reconstructed-sst

ERSST v5 specifications:
  - Grid: 2° × 2° (89 lat × 180 lon)
  - Coverage: 1854-present
  - Variables: SST (sea surface temperature)
  - Format: NetCDF (CF conventions)

Citation:
  Huang et al. (2017): NOAA Extended Reconstructed Sea Surface Temperature
  (ERSST), Version 5. doi:10.7289/V5T72FNM

Requirements
------------
* Packages: Downloads, NCDatasets, Dates
  Pkg.add(["Downloads", "NCDatasets", "Dates"])

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.0.1
Date: April 22, 2026
Assisted by: Claude Code (Anthropic)
=#

using Downloads
using Dates
using Printf

const BASE_URL = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/"
const DATA_DIR = joinpath(@__DIR__, "data")

"""
    download_ersst_year(year)

Download all monthly ERSST files for a given year.
Files are named: ersst.v5.YYYYMM.nc
"""
function download_ersst_year(year::Int)
    mkpath(DATA_DIR)

    files_downloaded = String[]

    for month in 1:12
        filename = @sprintf("ersst.v5.%04d%02d.nc", year, month)
        url = BASE_URL * filename
        outpath = joinpath(DATA_DIR, filename)

        if isfile(outpath)
            println("  $filename already exists, skipping")
            push!(files_downloaded, outpath)
            continue
        end

        print("  Downloading $filename...")
        try
            Downloads.download(url, outpath)
            println(" done")
            push!(files_downloaded, outpath)
        catch e
            println(" failed ($(typeof(e)))")
            # File might not exist yet for recent months
        end
    end

    return files_downloaded
end

"""
    download_ersst_range(start_year, end_year)

Download ERSST data for a range of years.
"""
function download_ersst_range(start_year::Int, end_year::Int)
    println("=" ^ 60)
    println("Downloading NOAA ERSST v5 data")
    println("  Years: $start_year - $end_year")
    println("  Source: $BASE_URL")
    println("=" ^ 60)

    all_files = String[]

    for year in start_year:end_year
        println("\nYear $year:")
        files = download_ersst_year(year)
        append!(all_files, files)
    end

    println("\n" * "=" ^ 60)
    println("Download complete: $(length(all_files)) files")
    println("=" ^ 60)

    return all_files
end

"""
    list_local_files()

List all downloaded ERSST files.
"""
function list_local_files()
    if !isdir(DATA_DIR)
        return String[]
    end
    files = filter(f -> endswith(f, ".nc"), readdir(DATA_DIR))
    return sort([joinpath(DATA_DIR, f) for f in files])
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    println("=" ^ 60)
    println("NOAA ERSST v5 Data Downloader")
    println("=" ^ 60)
    println("""
    This script downloads monthly Sea Surface Temperature data.

    Grid: 2° × 2° (89 lat × 180 lon = 16,020 ocean points)
    Time: Monthly, 1854-present (~2000 months)

    Each file is ~200KB, full dataset ~400MB.
    """)

    # Download recent decades for demo (1980-2023)
    # For full analysis, use 1854-present
    println("\nDownloading 1980-2023 (44 years, ~530 files)...")
    println("This will take a few minutes...\n")

    files = download_ersst_range(1980, 2023)

    println("\nTo download more years, run:")
    println("  download_ersst_range(1854, 1979)")

    # Show what we have
    local_files = list_local_files()
    if !isempty(local_files)
        println("\nLocal files: $(length(local_files))")
        println("  First: $(basename(first(local_files)))")
        println("  Last:  $(basename(last(local_files)))")
    end
end
