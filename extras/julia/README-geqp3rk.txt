$ gfortran -O2 -fopenmp dgeqp3rk_dr.f ~/scratch/tmp/lapack/liblapack.a ~/scratch/tmp/lapack/librefblas.a
$ a.out
        1024        2048
        1024        2048
 lwork=        6145
        1024  1019412004
  1024******  time =   0.1189
 info =         0
 K=          28
 REL_ERROR=   4.1114275718039301E-016
        2048        4096
        2048        4096
 lwork=       12289
        2048  1018256560
  2048******  time =   0.6076
 info =         0
 K=          31
 REL_ERROR=   1.8789354528421543E-016
        3072        6144
        3072        6144
 lwork=       18433
        3072  1018425297
  3072******  time =   1.3488
 info =         0
 K=          32
 REL_ERROR=   2.1574673373275703E-016
        4096        8192
        4096        8192
 lwork=       24577
        4096  1018530796
  4096******  time =   2.4456
 info =         0
 K=          33
 REL_ERROR=   2.3316126856365808E-016
Note: The following floating-point exceptions are signalling: IEEE_DENORMAL
$ gfortran -O2 -fopenmp dgeqp3rk_dr.f ~/scratch/tmp/lapack/liblapack.a -lblas
$ a.out
        1024        2048
        1024        2048
 lwork=        6145
        1024  1019414822
  1024******  time =   0.0283
 info =         0
 K=          28
 REL_ERROR=   4.1207356140965924E-016
        2048        4096
        2048        4096
 lwork=       12289
        2048  1018292413
  2048******  time =   0.1493
 info =         0
 K=          31
 REL_ERROR=   1.9381408019268171E-016
        3072        6144
        3072        6144
 lwork=       18433
        3072  1018393335
  3072******  time =   0.2264
 info =         0
 K=          32
 REL_ERROR=   2.1046905642584424E-016
        4096        8192
        4096        8192
 lwork=       24577
        4096  1018518804
  4096******  time =   0.4532
 info =         0
 K=          33
 REL_ERROR=   2.3118129311619213E-016
Note: The following floating-point exceptions are signalling: IEEE_DENORMAL


$ gfortran -O2 -fopenmp zgeqp3rk_dr.f ~/scratch/tmp/lapack/liblapack.a ~/scratch/tmp/lapack/librefblas.a
$ a.out
        1024        2048
        1024        2048
 lwork=        6145
        1024  1019415053
  1024******  time =   0.1988
 info =         0
 K=          28
 REL_ERROR=   4.1214990083616635E-016
        2048        4096
        2048        4096
 lwork=       12289
        2048  1018266260
  2048******  time =   0.9677
 info =         0
 K=          31
 REL_ERROR=   1.8949535275750446E-016
        3072        6144
        3072        6144
 lwork=       18433
        3072  1018406380
  3072******  time =   2.2126
 info =         0
 K=          32
 REL_ERROR=   2.1262302304311105E-016
        4096        8192
        4096        8192
 lwork=       24577
        4096  1018520129
  4096******  time =   4.0628
 info =         0
 K=          33
 REL_ERROR=   2.3140006196957595E-016
Note: The following floating-point exceptions are signalling: IEEE_DENORMAL
$ gfortran -O2 -fopenmp zgeqp3rk_dr.f ~/scratch/tmp/lapack/liblapack.a -lblas
$ a.out
        1024        2048
        1024        2048
 lwork=        6145
        1024  1019415493
  1024******  time =   0.0353
 info =         0
 K=          28
 REL_ERROR=   4.1229521644271595E-016
        2048        4096
        2048        4096
 lwork=       12289
        2048  1018292146
  2048******  time =   0.1790
 info =         0
 K=          31
 REL_ERROR=   1.9377000855564564E-016
        3072        6144
        3072        6144
 lwork=       18433
        3072  1018393568
  3072******  time =   0.4387
 info =         0
 K=          32
 REL_ERROR=   2.1050752908885828E-016
        4096        8192
        4096        8192
 lwork=       24577
        4096  1018539393
  4096******  time =   0.7801
 info =         0
 K=          33
 REL_ERROR=   2.3458083280183318E-016
Note: The following floating-point exceptions are signalling: IEEE_DENORMAL
