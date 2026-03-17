# Gotchas in Calculating EOFs

Common pitfalls and issues when performing Empirical Orthogonal Function (EOF) analysis.

## Data Preprocessing

1. **Area weighting** - For gridded data, you must weight by the square root of cos(latitude) or grid cell area. Without this, high-latitude points are overweighted since grid cells converge toward the poles.

2. **Mean removal** - Ambiguity about whether to remove the temporal mean at each grid point (most common), the spatial mean at each time step, or the global mean. Usually you want the temporal mean removed.

3. **Standardization choice** - Using the covariance matrix (preserves variance differences) vs correlation matrix (standardizes each location). The choice significantly affects results.

4. **Missing data** - EOFs require complete matrices. Infilling, masking, or iterative methods each introduce biases.

## Interpretation Pitfalls

5. **Sign ambiguity** - EOFs are only defined up to a sign flip (eigenvectors can be ±1). You must choose a convention (e.g., positive loading in a reference region).

6. **Orthogonality constraint** - EOFs are mathematically orthogonal, but physical climate modes aren't. Higher EOFs often show artificial dipole structures (Buell patterns) due to this constraint.

7. **Domain dependence** - Results change with the spatial domain you analyze. EOFs aren't intrinsic to the data—they depend on your analysis region.

8. **Variance ≠ physical significance** - EOFs maximize explained variance, not physical meaningfulness. A mode explaining 30% of variance isn't necessarily more "real" than one explaining 5%.

## Statistical Issues

9. **North's rule of thumb** - Eigenvalues within √(2/N) of each other are effectively degenerate. Their EOFs can mix and rotate arbitrarily. Don't over-interpret individual patterns when eigenvalues are close.

10. **Sampling error** - With limited samples, trailing EOFs are mostly noise. The number of meaningful EOFs is typically much smaller than min(space, time).

## Computational

11. **Centering before SVD** - If using SVD, the data must be centered first. Some implementations don't do this automatically.

12. **Which decomposition** - Decomposing the (space × space) covariance matrix vs using SVD on the (time × space) anomaly matrix should give identical EOFs but may differ numerically with ill-conditioned data.
