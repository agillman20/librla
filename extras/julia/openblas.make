all: double dcomplex single scomplex

openblas=/Users/gimbutas/aarch64/OpenBLAS


double:
	gfortran -o ./dgeqp3rk_openblas -O3 -fdefault-integer-8 dgeqp3rk_dr.f -fopenmp -L$(openblas)  -lopenblas
	./dgeqp3rk_openblas

single:
	gfortran -o ./sgeqp3rk_openblas -O3 -fdefault-integer-8 sgeqp3rk_dr.f -fopenmp -L$(openblas)  -lopenblas
	./sgeqp3rk_openblas

dcomplex:
	gfortran -o ./zgeqp3rk_openblas -O3 -fdefault-integer-8 zgeqp3rk_dr.f -fopenmp -L$(openblas)  -lopenblas
	./zgeqp3rk_openblas

scomplex:
	gfortran -o ./cgeqp3rk_openblas -O3 -fdefault-integer-8 cgeqp3rk_dr.f -fopenmp -L$(openblas)  -lopenblas
	./cgeqp3rk_openblas

