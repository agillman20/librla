
$ gfortran -O3 -fopenmp dgeqp3rk_dr.f /Users/gimbutas/aarch64/lapack-ref/liblapack.a /Users/gimbutas/aarch64/lapack-ref/librefblas.a
$ a.out
*****  time =   0.0735   GFLOPS=************
 info =         0
 K=          23
 REL_ERROR=   3.3570482335892176E-013
*****  time =   0.2244   GFLOPS=************
 info =         0
 K=          25
 REL_ERROR=   6.0272686977435950E-013


$ gfortran -O3 -fopenmp zgeqp3rk_dr.f /Users/gimbutas/aarch64/lapack-ref/liblapack.a /Users/gimbutas/aarch64/lapack-ref/librefblas.a
$ a.out
*****  time =   0.3505   GFLOPS=************
 info =         0
 K=        1024
 REL_ERROR=   0.0000000000000000     
*****  time =   2.6247   GFLOPS=************
 info =         0
 K=        2048
 REL_ERROR=   0.0000000000000000     

