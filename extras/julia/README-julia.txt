julia> include("libid_lapack_dr.jl")
Float64
(2000, 4000)
info=0
cmplx=false
lwork=12001
info=0
  0.079786 seconds (142.97 k allocations: 68.354 MiB, 0.73% gc time, 56.12% compilation time)
  0.415856 seconds (2.87 M allocations: 209.103 MiB, 6.65% gc time, 91.93% compilation time)
k=30
err=1.22756386790157e-13
  0.006863 seconds (13.40 k allocations: 1.674 MiB, 96.05% compilation time)
abserr=9.163047210423488e-15
ComplexF64
(4000, 2000)
info=0
cmplx=true
lwork=6001
info=0
  0.051828 seconds (90.68 k allocations: 126.778 MiB, 4.60% gc time, 56.34% compilation time)
  0.289186 seconds (2.05 M allocations: 225.407 MiB, 2.37% gc time, 95.28% compilation time)
k=0
err=4.18871174466815


julia> include("libid_dr.jl")
Float64
(2000, 4000)
if_randomized=0
=======================
  0.337464 seconds (1.97 M allocations: 161.601 MiB, 0.58% gc time, 62.20% compilation time)
  0.132702 seconds (1.51 k allocations: 61.104 MiB, 3.28% gc time)
k=31
=== house_decom_piv ===
  0.135663 seconds (3.51 k allocations: 133.375 KiB)
k=31
  0.048807 seconds (326.56 k allocations: 17.280 MiB, 98.69% compilation time)
abserr=3.0308046194492037e-15
abserr=3.0308046194492037e-15
  0.000432 seconds (23 allocations: 961.984 KiB)
abserr=3.0308046194492037e-15
abserr=3.0308046194492037e-15


