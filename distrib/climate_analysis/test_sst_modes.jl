#=
test_sst_modes.jl - Modal decomposition of raw NOAA SST data using SVD.

Description
-----------
Applies SVD/EOF analysis to raw Sea Surface Temperature data (no anomaly
preprocessing). The dominant modes capture:

  1. EOF1: Mean spatial pattern (warm tropics, cold poles)
  2. EOF2-4: Seasonal variations
  3. EOF5: ENSO (El Niño-Southern Oscillation)

This demonstrates randomized SVD (svd_sketch) on real climate data.

Data: Raw SST values in °C (no anomaly computation, no detrending)

EOF (Empirical Orthogonal Functions) = PCA = POD = SVD

Prerequisites
-------------
Run download_sst_data.jl first:
  julia download_sst_data.jl

Requirements
------------
* librla.jl in the path
* Packages: NCDatasets, Statistics, Dates, FFTW, GLMakie

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.0.1
Date: April 22, 2026
Assisted by: Claude Code (Anthropic)
=#

using LinearAlgebra
using Statistics
using Dates
using FFTW
using NCDatasets
using GLMakie
using GLFW
using Printf

# Include librla from parent directory
include(joinpath(@__DIR__, "..", "julia", "librla.jl"))
using .librla

# Include local utilities
include(joinpath(@__DIR__, "test_sst_utils.jl"))
using .TestSstUtils

# ========================================================================
# Load SST data
# ========================================================================

println("=" ^ 70)
println("NOAA SST Climate Mode Analysis (EOF/SVD)")
println("=" ^ 70)

const DATA_DIR = joinpath(@__DIR__, "data")

# Find all downloaded files
nc_files = sort(filter(f -> endswith(f, ".nc"), readdir(DATA_DIR, join=true)))

if isempty(nc_files)
    error("""
    No data files found in $DATA_DIR

    Please run download_sst_data.jl first:
      julia download_sst_data.jl
    """)
end

println("\nFound $(length(nc_files)) monthly SST files")
println("  First: $(basename(first(nc_files)))")
println("  Last:  $(basename(last(nc_files)))")

# Load first file to get grid info
ds = NCDataset(first(nc_files))
lon = ds["lon"][:]
lat = ds["lat"][:]
close(ds)

n_lon = length(lon)
n_lat = length(lat)
n_time = length(nc_files)

println("\nGrid dimensions:")
println("  Longitude: $n_lon points ($(lon[1])° to $(lon[end])°)")
println("  Latitude:  $n_lat points ($(lat[1])° to $(lat[end])°)")
println("  Time:      $n_time months")

# Load all SST data
println("\nLoading SST data...")
SST = zeros(Float32, n_lon, n_lat, n_time)
dates = Date[]

for (i, f) in enumerate(nc_files)
    local ds = NCDataset(f)
    # SST data may contain Missing values - convert to NaN
    sst_raw = ds["sst"][:, :, 1, 1]  # lon × lat × lev × time
    SST[:, :, i] = coalesce.(sst_raw, NaN32)

    # Extract date from filename (ersst.v5.YYYYMM.nc)
    m = match(r"ersst\.v5\.(\d{4})(\d{2})\.nc", basename(f))
    if m !== nothing
        push!(dates, Date(parse(Int, m[1]), parse(Int, m[2]), 1))
    end
    close(ds)

    if i % 100 == 0
        print("  Loaded $i / $n_time\r")
    end
end
println("  Loaded $n_time months              ")

# Handle missing values (land = NaN in ERSST)
SST[SST .< -900] .= NaN  # Missing value flag

# Center on Greenwich meridian: convert 0-360 to -180 to 180
println("\nCentering on Greenwich meridian...")
shift_idx = findfirst(lon .>= 180)
lon = vcat(lon[shift_idx:end] .- 360, lon[1:shift_idx-1])
SST = vcat(SST[shift_idx:end, :, :], SST[1:shift_idx-1, :, :])

println("  Longitude: $(lon[1])° to $(lon[end])°")
println("\nSST range: $(round(minimum(filter(!isnan, SST)), digits=1))°C to $(round(maximum(filter(!isnan, SST)), digits=1))°C")

# ========================================================================
# Create ocean mask and reshape for SVD
# ========================================================================

println("\n" * "=" ^ 70)
println("Preparing data matrix...")
println("=" ^ 70)

# Ocean mask (where we have valid data for all times)
ocean_mask = .!any(isnan.(SST), dims=3)[:, :, 1]
n_ocean = sum(ocean_mask)

println("\n  Ocean points: $n_ocean / $(n_lon * n_lat) ($(round(100*n_ocean/(n_lon*n_lat), digits=1))%)")
println("  Area weighting: None (unweighted EOF analysis)")

# Reshape to 2D matrix: ocean_points × time
# Each column is one month's SST field (ocean points only)
SST_matrix = zeros(Float32, n_ocean, n_time)
ocean_idx = findall(ocean_mask)

for t in 1:n_time
    sst_t = SST[:, :, t]
    SST_matrix[:, t] = sst_t[ocean_idx]
end

println("  Data matrix: $n_ocean × $n_time")

# ========================================================================
# EOF Analysis via SVD
# ========================================================================

println("\n" * "=" ^ 70)
println("EOF Analysis (SVD)")
println("=" ^ 70)

# Number of modes to compute
n_modes = 30

println("\nComputing $(n_modes)-mode SVD using svd_sketch...")
t0 = time()
U, σ, Vt = librla.svd_sketch(SST_matrix, n_modes, power_iter=2, extra_samples=10)
V = Vt'
println("  Elapsed time: $(round(time() - t0, digits=2))s")

# Also compute a few more modes with full SVD for comparison
println("\nComputing reference SVD...")
t0 = time()
U_full, σ_full, V_full = svd(SST_matrix)
println("  Elapsed time: $(round(time() - t0, digits=2))s")

# Variance explained
total_var = sum(σ_full.^2)
var_explained = σ_full.^2 / total_var
cumulative_var = cumsum(var_explained)

println("\nEOF variance explained:")
println("  Mode    Variance %   Cumulative %")
println("  " * "-" ^ 40)
for i in 1:min(10, n_modes)
    @printf("  EOF%2d   %6.2f%%      %6.2f%%\n",
            i, 100*var_explained[i], 100*cumulative_var[i])
end

n90 = findfirst(cumulative_var .>= 0.90)
n95 = findfirst(cumulative_var .>= 0.95)
println("\n  Modes for 90%: $n90, 95%: $n95")

# North's rule of thumb for mode separability
print_north_test(σ_full, n_time, n_modes=10)

# ========================================================================
# Reshape EOFs back to spatial maps
# ========================================================================

function reshape_to_map(vec, mask, lon, lat)
    field = fill(NaN32, length(lon), length(lat))
    field[mask] .= vec
    return field
end

# Compute Niño 3.4 index for ENSO validation (not for sign fixing)
# Niño 3.4 region: 5°N-5°S, 170°W-120°W (-170° to -120° in -181.0.0 system)
nino34_lon_idx = findall(x -> -170 <= x <= -120, lon)
nino34_lat_idx = findall(x -> -5 <= x <= 5, lat)

nino34_index = zeros(n_time)
for t in 1:n_time
    region = SST[nino34_lon_idx, nino34_lat_idx, t]
    nino34_index[t] = mean(filter(!isnan, region))
end

# Fix signs based on spatial pattern mean
# Convention: spatial pattern should have positive mean over ocean
# This ensures positive PC = positive anomaly contribution

function fix_svd_signs!(U_modes, V_modes)
    for i in 1:size(U_modes, 2)
        if mean(U_modes[:, i]) < 0
            U_modes[:, i] .*= -1
            V_modes[:, i] .*= -1
        end
    end
end

# Deterministic SVD
U_det = copy(U_full[:, 1:n_modes])
V_det = copy(V_full[:, 1:n_modes])
fix_svd_signs!(U_det, V_det)

# Randomized SVD
U_rand = copy(U)
V_rand = copy(V)
fix_svd_signs!(U_rand, V_rand)

# Spatial: U (non-dimensional pattern)
# Temporal: projection of data onto U, normalized by √n_ocean for °C units
EOF_maps_det = [reshape_to_map(U_det[:, i], ocean_mask, lon, lat) for i in 1:n_modes]
PC_det = [(U_det[:, i]' * SST_matrix)' / sqrt(n_ocean) for i in 1:n_modes]

EOF_maps_rand = [reshape_to_map(U_rand[:, i], ocean_mask, lon, lat) for i in 1:n_modes]
PC_rand = [(U_rand[:, i]' * SST_matrix)' / sqrt(n_ocean) for i in 1:n_modes]

# ========================================================================
# Visualize Deterministic SVD results
# ========================================================================

println("\n" * "=" ^ 70)
println("Visualizing EOF patterns...")
println("=" ^ 70)

# Convert dates to decimal years for plotting
years = [Dates.year(d) + (Dates.month(d) - 1)/12 for d in dates]

# Mode labels
mode_names = ["EOF1", "EOF2", "EOF3", "EOF4", "EOF5"]

fig1 = Figure(size=(1400, 1000))
Label(fig1[0, :], "Deterministic SVD", fontsize=20, font=:bold)

for i in 1:5
    # Left: Spatial EOF pattern
    ax_spatial = GLMakie.Axis(fig1[i, 1], xlabel="Longitude", ylabel="Latitude",
                              title="$(mode_names[i]) ($(round(100*var_explained[i], digits=1))%)",
                              aspect=DataAspect())
    clim = maximum(filter(!isnan, abs.(EOF_maps_det[i]))) * 0.8
    heatmap!(ax_spatial, lon, lat, EOF_maps_det[i], colormap=Reverse(:RdBu),
             colorrange=(-clim, clim))

    # Right: Temporal PC coefficient (auto-scale y-axis to fit data)
    ax_temporal = GLMakie.Axis(fig1[i, 2], xlabel="Year", ylabel="PC$i")
    lines!(ax_temporal, years, PC_det[i], linewidth=1, color=:steelblue)
    if i > 1  # Only show zero line for oscillating modes, not PC1
        hlines!(ax_temporal, [0], color=:gray, linestyle=:dash)
    end
end

colsize!(fig1.layout, 1, Relative(0.45))
colsize!(fig1.layout, 2, Relative(0.55))

scr1 = display(GLMakie.Screen(title="Figure 1 - Deterministic SVD"), fig1)
GLFW.SetWindowPos(scr1.glscreen, 50, 50)

# ========================================================================
# Visualize Randomized SVD results
# ========================================================================

# Variance explained by randomized modes
var_explained_rand = σ.^2 / total_var

fig2 = Figure(size=(1400, 1000))
Label(fig2[0, :], "Randomized SVD", fontsize=20, font=:bold)

for i in 1:5
    # Left: Spatial EOF pattern
    ax_spatial = GLMakie.Axis(fig2[i, 1], xlabel="Longitude", ylabel="Latitude",
                              title="$(mode_names[i]) ($(round(100*var_explained_rand[i], digits=1))%)",
                              aspect=DataAspect())
    clim = maximum(filter(!isnan, abs.(EOF_maps_rand[i]))) * 0.8
    heatmap!(ax_spatial, lon, lat, EOF_maps_rand[i], colormap=Reverse(:RdBu),
             colorrange=(-clim, clim))

    # Right: Temporal PC coefficient (auto-scale y-axis to fit data)
    ax_temporal = GLMakie.Axis(fig2[i, 2], xlabel="Year", ylabel="PC$i")
    lines!(ax_temporal, years, PC_rand[i], linewidth=1, color=:steelblue)
    if i > 1  # Only show zero line for oscillating modes, not PC1
        hlines!(ax_temporal, [0], color=:gray, linestyle=:dash)
    end
end

colsize!(fig2.layout, 1, Relative(0.45))
colsize!(fig2.layout, 2, Relative(0.55))

scr2 = display(GLMakie.Screen(title="Figure 2 - Randomized SVD"), fig2)
GLFW.SetWindowPos(scr2.glscreen, 100, 100)

# ========================================================================
# Singular value spectrum and variance comparison
# ========================================================================

fig3 = Figure(size=(600, 400))

# Singular value spectrum
ax3 = GLMakie.Axis(fig3[1,1], xlabel="Mode", ylabel="σᵢ",
                   title="Singular Value Spectrum", yscale=log10)
scatter!(ax3, 1:min(50, length(σ_full)), σ_full[1:min(50, length(σ_full))],
         markersize=6, label="Deterministic")
scatter!(ax3, 1:n_modes, σ, markersize=10, marker=:cross, color=:red,
         label="Randomized")
axislegend(ax3, position=:rt)

scr3 = display(GLMakie.Screen(title="Figure 3 - Singular Values"), fig3)
GLFW.SetWindowPos(scr3.glscreen, 150, 150)

# ========================================================================
# ENSO region analysis
# ========================================================================

println("\n" * "=" ^ 70)
println("ENSO Analysis")
println("=" ^ 70)

# Simple linear detrend function
function detrend(x)
    n = length(x)
    t = 1:n
    slope = cor(t, x) * std(x) / std(t)
    intercept = mean(x) - slope * mean(t)
    return x .- (intercept .+ slope .* t)
end

# Niño 3.4 index was computed earlier for sign determination
# Correlation without detrending
corr_pc5_nino_raw = cor(PC_det[5], nino34_index)
println("\nCorrelation between PC5 and Niño 3.4 index: $(round(corr_pc5_nino_raw, digits=3))")

# Detrend to remove warming trend and reveal ENSO signal
pc5_detrend = detrend(PC_det[5])
nino34_detrend = detrend(nino34_index)

# Correlation between detrended PC5 (ENSO mode) and Niño 3.4
corr_pc5_nino = cor(pc5_detrend, nino34_detrend)
println("Correlation (detrended): $(round(corr_pc5_nino, digits=3))")

# Plot comparison (detrended and normalized)
fig4 = Figure(size=(900, 400))
ax4 = GLMakie.Axis(fig4[1,1], xlabel="Year", ylabel="Index (detrended, normalized)",
                   title="PC5 (ENSO) vs Niño 3.4 Index (r = $(round(corr_pc5_nino, digits=2)), detrended)")
lines!(ax4, years, pc5_detrend / std(pc5_detrend), linewidth=1.5, label="PC5 (det)", color=:blue)
lines!(ax4, years, nino34_detrend / std(nino34_detrend), linewidth=1.5,
       label="Niño 3.4", color=:red, linestyle=:dash)
hlines!(ax4, [0], color=:gray, linestyle=:dot)
axislegend(ax4, position=:lt)

scr4 = display(GLMakie.Screen(title="Figure 4 - ENSO Comparison"), fig4)
GLFW.SetWindowPos(scr4.glscreen, 200, 200)

# ========================================================================
# Summary
# ========================================================================

println("\n" * "=" ^ 70)
println("Summary")
println("=" ^ 70)
println("  Data: NOAA ERSST v5")
println("  Grid: $n_lon × $n_lat ($(round(2, digits=0))° resolution)")
println("  Time: $(dates[1]) to $(dates[end]) ($n_time months)")
println("  Ocean points: $n_ocean")
println("")
println("  EOF Analysis:")
println("    EOF1 variance: $(round(100*var_explained[1], digits=1))% (mean pattern)")
println("    EOF5 variance: $(round(100*var_explained[5], digits=1))% (ENSO)")
println("    Modes for 90% variance: $n90")
separable = north_test(σ_full, n_time)
n_well_separated = sum(separable[1:10])
println("    Well-separated modes (North's rule): $n_well_separated/10")
println("")
println("  Randomized SVD accuracy:")
rel_err = norm(σ - σ_full[1:n_modes]) / norm(σ_full[1:n_modes])
println("    Singular value error: $(round(100*rel_err, digits=3))%")
println("")
println("  ENSO validation (PC5 - Niño 3.4 correlation):")
pc5_rand_detrend = detrend(PC_rand[5])
corr_pc5_rand_raw = cor(PC_rand[5], nino34_index)
corr_pc5_rand = cor(pc5_rand_detrend, nino34_detrend)
println("    Deterministic SVD: $(round(corr_pc5_nino_raw, digits=3)) (detrended: $(round(corr_pc5_nino, digits=3)))")
println("    Randomized SVD:    $(round(corr_pc5_rand_raw, digits=3)) (detrended: $(round(corr_pc5_rand, digits=3)))")
println("=" ^ 70)
