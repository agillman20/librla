# Attribution and Code Similarity Report

This document records the results of a code similarity check performed on 2025-12-16
to ensure proper attribution for the climate analysis examples.

## download_sst_data.py / .jl / .m

**Finding:** The code appears to be **original**. No exact matches found.

- The download URL pattern (`https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/ersst.v5.YYYYMM.nc`) is the official NOAA source
- Most similar code uses `wget` or xarray OPeNDAP, not Python `requests`
- No GitHub repositories found with similar `download_ersst_year()` / `download_ersst_range()` function structure

## test_sst_anomaly.jl / test_sst_modes.jl / .py / .m

**Finding:** The *methodology* (EOF/SVD on SST) is a **well-established climate analysis technique**. Several similar implementations exist:

| Repository | Similarity | Notes |
|------------|------------|-------|
| [pangeo-data/pangeo-ocean-examples](https://github.com/pangeo-data/pangeo-ocean-examples/blob/master/noaa_ersst_variability.ipynb) | High conceptual | Same dataset, same analysis (EOF of ERSST, Nino3.4), uses `eofs` package |
| [royalosyin/Python-Practical-Application-on-Climate-Variability-Studies](https://github.com/royalosyin/Python-Practical-Application-on-Climate-Variability-Studies/blob/master/ex18-EOF%20analysis%20global%20SST.ipynb) | Medium | EOF on global SST, similar preprocessing |
| [PO.DAAC Cookbook](https://podaac.github.io/tutorials/notebooks/DataStories/eof_example_ersst.html) | Medium | ERSST EOF analysis, uses `xeofs` package |
| [eofs library](https://ajdawson.github.io/eofs/) | Conceptual | Standard Python EOF library |

## Data Source Attribution

The climate analysis examples use NOAA ERSST v5 data:

**Citation:**
> Huang et al. (2017): NOAA Extended Reconstructed Sea Surface Temperature (ERSST), Version 5.
> NOAA National Centers for Environmental Information. doi:10.7289/V5T72FNM

**Data source:** https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/

**Product info:** https://www.ncei.noaa.gov/products/extended-reconstructed-sst

## Algorithm References

The randomized SVD methodology is based on:

> Halko, Martinsson, Tropp. "Finding structure with randomness: Probabilistic algorithms
> for constructing approximate matrix decompositions." SIAM Review, 2011.

## Conclusion

The implementation code is original, developed with assistance from Claude Code (Anthropic).
The methodology follows standard EOF/SVD climate analysis practices as described in the
references above.
