to compile the blas call: mex -v -R2017b matrixMultiply.c -lmwblas

Example running it: 
A = [1 3 5; 2 4 7];
B = [-5 8 11; 3 9 21; 4 0 8];
X = matrixMultiply(A,B)
