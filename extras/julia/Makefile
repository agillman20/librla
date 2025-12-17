all: double dcomplex single scomplex

reflapack=~/scratch/tmp/lapack


double:
	gfortran -o ./dgeqp3rk_ref -O2 -fopenmp dgeqp3rk_dr.f $(reflapack)/liblapack.a $(reflapack)/librefblas.a
	./dgeqp3rk_ref
	gfortran -o ./dgeqp3rk_blas -O2 -fopenmp dgeqp3rk_dr.f $(reflapack)/liblapack.a -lblas
	./dgeqp3rk_blas


single:
	gfortran -o ./sgeqp3rk_ref -O2 -fopenmp sgeqp3rk_dr.f $(reflapack)/liblapack.a $(reflapack)/librefblas.a
	./sgeqp3rk_ref
	gfortran -o ./sgeqp3rk_blas -O2 -fopenmp sgeqp3rk_dr.f $(reflapack)/liblapack.a -lblas
	./sgeqp3rk_blas


dcomplex:
	gfortran -o ./zgeqp3rk_ref -O2 -fopenmp zgeqp3rk_dr.f $(reflapack)/liblapack.a $(reflapack)/librefblas.a
	./zgeqp3rk_ref
	gfortran -o ./zgeqp3rk_blas -O2 -fopenmp zgeqp3rk_dr.f $(reflapack)/liblapack.a -lblas
	./zgeqp3rk_blas


scomplex:
	gfortran -o ./cgeqp3rk_ref -O2 -fopenmp cgeqp3rk_dr.f $(reflapack)/liblapack.a $(reflapack)/librefblas.a
	./cgeqp3rk_ref
	gfortran -o ./cgeqp3rk_blas -O2 -fopenmp cgeqp3rk_dr.f $(reflapack)/liblapack.a -lblas
	./cgeqp3rk_blas
