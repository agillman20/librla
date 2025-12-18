gfortran -o ./dgeqp3rk_openblas -O3 -fdefault-integer-8 dgeqp3rk_dr.f -fopenmp -L/Users/gimbutas/aarch64/OpenBLAS  -lopenblas
./dgeqp3rk_openblas
                 1024                 2048
                 1024                 2048
 lwork=                 6145
                 1024                 2048
  1024  2048  time =   0.0163
 info =         0
 K=                   28
 REL_ERROR=   4.1214517330754826E-016
                 2048                 4096
                 2048                 4096
 lwork=                12289
                 2048                 4096
  2048  4096  time =   0.0872
 info =         0
 K=                   31
 REL_ERROR=   1.8814076614025490E-016
                 3072                 6144
                 3072                 6144
 lwork=                18433
                 3072                 6144
  3072  6144  time =   0.5265
 info =         0
 K=                   32
 REL_ERROR=   2.1284127151428908E-016
                 4096                 8192
                 4096                 8192
 lwork=                24577
                 4096                 8192
  4096  8192  time =   0.7844
 info =         0
 K=                   33
 REL_ERROR=   2.3619015070563211E-016
gfortran -o ./zgeqp3rk_openblas -O3 -fdefault-integer-8 zgeqp3rk_dr.f -fopenmp -L/Users/gimbutas/aarch64/OpenBLAS  -lopenblas
./zgeqp3rk_openblas
                 1024                 2048
                 1024                 2048
 lwork=                 6145
                 1024                 2048
  1024  2048  time =   0.0326
 info =         0
 K=                   28
 REL_ERROR=   4.1157638881934697E-016
                 2048                 4096
                 2048                 4096
 lwork=                12289
                 2048                 4096
  2048  4096  time =   0.2102
 info =         0
 K=                   31
 REL_ERROR=   1.8955742784977963E-016
                 3072                 6144
                 3072                 6144
 lwork=                18433
                 3072                 6144
  3072  6144  time =   0.2411
 info =         0
 K=                   32
 REL_ERROR=   2.1249602591527192E-016
                 4096                 8192
                 4096                 8192
 lwork=                24577
                 4096                 8192
  4096  8192  time =   0.3950
 info =         0
 K=                   33
 REL_ERROR=   2.3053549457111049E-016
gfortran -o ./sgeqp3rk_openblas -O3 -fdefault-integer-8 sgeqp3rk_dr.f -fopenmp -L/Users/gimbutas/aarch64/OpenBLAS  -lopenblas
./sgeqp3rk_openblas
                 1024                 2048
                 1024                 2048
 lwork=                 6145
                 1024                 2048
  1024  2048  time =   0.0159
 info =         0
 K=                   14
 REL_ERROR=   8.06743117E-08
                 2048                 4096
                 2048                 4096
 lwork=                12289
                 2048                 4096
  2048  4096  time =   0.0419
 info =         0
 K=                   16
 REL_ERROR=   1.88119547E-08
                 3072                 6144
                 3072                 6144
 lwork=                18433
                 3072                 6144
  3072  6144  time =   0.0947
 info =         0
 K=                   16
 REL_ERROR=   4.98543216E-08
                 4096                 8192
                 4096                 8192
 lwork=                24577
                 4096                 8192
  4096  8192  time =   0.2045
 info =         0
 K=                   16
 REL_ERROR=   7.44274544E-08
gfortran -o ./cgeqp3rk_openblas -O3 -fdefault-integer-8 cgeqp3rk_dr.f -fopenmp -L/Users/gimbutas/aarch64/OpenBLAS  -lopenblas
./cgeqp3rk_openblas
                 1024                 2048
                 1024                 2048
 lwork=                 6145
                 1024                 2048
  1024  2048  time =   0.0178
 info =         0
 K=                   14
 REL_ERROR=   8.46101003E-08
                 2048                 4096
                 2048                 4096
 lwork=                12289
                 2048                 4096
  2048  4096  time =   0.0500
 info =         0
 K=                   16
 REL_ERROR=   1.92033198E-08
                 3072                 6144
                 3072                 6144
 lwork=                18433
                 3072                 6144
  3072  6144  time =   0.1030
 info =         0
 K=                   16
 REL_ERROR=   4.45186821E-08
                 4096                 8192
                 4096                 8192
 lwork=                24577
                 4096                 8192
  4096  8192  time =   0.2165
 info =         0
 K=                   16
 REL_ERROR=   7.71307143E-08
