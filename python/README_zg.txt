    
If you wish to install a Python library that isn't in Homebrew,
use a virtual environment:
    
python3 -m venv path/to/venv
source path/to/venv/bin/activate
python3 -m pip install xyz
    
If you wish to install a Python application that isn't in Homebrew,
it may be easiest to use 'pipx install xyz', which will manage a
virtual environment for you. You can install pipx with
    
brew install pipx
    
You may restore the old behavior of pip by passing
the '--break-system-packages' flag to pip, or by adding
'break-system-packages = true' to your pip.conf file. The latter
will permanently disable this error.
    
If you disable this error, we STRONGLY recommend that you additionally
pass the '--user' flag to pip, or set 'user = true' in your pip.conf
file. Failure to do this can result in a broken Homebrew installation.
    
Read more about this behavior here: <https://peps.python.org/pep-0668/>

-------------------------------------------------------------------------------

Installation:

python3 -m venv local
source local/bin/activate

###python3 -m pip install xyz
pip3 install jax

-------------------------------------------------------------------------------

activate/deactivate:

$ source local/bin/activate
(local) $ deactivate 
$ 

-------------------------------------------------------------------------------

machdep.cpu.brand_string: Apple M4 Max

$ source local/bin/activate
(local) $ python rrqr_rand1_dr.py 
a shape (4000, 2000)
Elapsed time: 0.0914 seconds
Elapsed time: 0.0074 seconds
Elapsed time: 0.0072 seconds
Elapsed time: 0.0072 seconds
Elapsed time: 0.0072 seconds
Elapsed time: 0.0072 seconds
Elapsed time: 0.0076 seconds
Elapsed time: 0.0074 seconds
Elapsed time: 0.0072 seconds
r.shape =  (31, 2000)
k = 31
relerr =  2.271840641809376e-15
(local) $ python rrqr_rand2_dr.py 
a shape (4000, 2000)
Elapsed time: 0.0100 seconds
Elapsed time: 0.0079 seconds
Elapsed time: 0.0067 seconds
Elapsed time: 0.0077 seconds
Elapsed time: 0.0070 seconds
Elapsed time: 0.0073 seconds
Elapsed time: 0.0067 seconds
Elapsed time: 0.0065 seconds
Elapsed time: 0.0064 seconds
r.shape =  (31, 2000)
k = 31
relerr =  3.864282280129504e-15
(local) $ deactivate 

-------------------------------------------------------------------------------

model name	: Intel(R) Xeon(R) Gold 6254 CPU @ 3.10GHz
stepping	: 7
microcode	: 0x5003707
cpu MHz		: 3100.000
cache size	: 25344 KB
physical id	: 0
siblings	: 36
core id		: 27
cpu cores	: 18

Blas libraries using hyperthreading might need
OMP_NUM_THREADS=num_cpu_cores, in order to run properly:

OMP_NUM_THREADS=18 python3 rrqr_rand2_dr.py

$ source local/bin/activate
(local) $ python rrqr_rand2_dr.py 
a shape (4000, 2000)
Elapsed time: 0.3172 seconds
Elapsed time: 0.3636 seconds
Elapsed time: 0.3916 seconds
Elapsed time: 0.3763 seconds
Elapsed time: 0.4122 seconds
Elapsed time: 0.3557 seconds
Elapsed time: 0.3959 seconds
Elapsed time: 0.4305 seconds
Elapsed time: 0.2812 seconds
r.shape =  (31, 2000)
k = 31
relerr =  1.5743179148173928e-15
Elapsed time: 0.2426 seconds
Elapsed time: 0.2536 seconds
Elapsed time: 0.2415 seconds
Elapsed time: 0.1940 seconds
Elapsed time: 0.2049 seconds
Elapsed time: 0.1947 seconds
Elapsed time: 0.2064 seconds
Elapsed time: 0.1939 seconds
Elapsed time: 0.2052 seconds
Flam k= 30
Flam relerr =  5.375315566989096e-15
(local) $ OMP_NUM_THREADS=18 python3 rrqr_rand2_dr.py 
a shape (4000, 2000)
Elapsed time: 0.0275 seconds
Elapsed time: 0.0178 seconds
Elapsed time: 0.0207 seconds
Elapsed time: 0.0176 seconds
Elapsed time: 0.0162 seconds
Elapsed time: 0.0169 seconds
Elapsed time: 0.0173 seconds
Elapsed time: 0.0163 seconds
Elapsed time: 0.0175 seconds
r.shape =  (31, 2000)
k = 31
relerr =  1.5216784269534164e-15
Elapsed time: 0.2362 seconds
Elapsed time: 0.2415 seconds
Elapsed time: 0.2361 seconds
Elapsed time: 0.1944 seconds
Elapsed time: 0.2065 seconds
Elapsed time: 0.1946 seconds
Elapsed time: 0.2051 seconds
Elapsed time: 0.1928 seconds
Elapsed time: 0.2053 seconds
Flam k= 31
Flam relerr =  1.521396124457959e-15
(local) $ deactivate


Using the system python:

$ python3 rrqr_rand2_dr.py 
a shape (4000, 2000)
Elapsed time: 0.0391 seconds
Elapsed time: 0.0263 seconds
Elapsed time: 0.0199 seconds
Elapsed time: 0.0196 seconds
Elapsed time: 0.0216 seconds
Elapsed time: 0.0183 seconds
Elapsed time: 0.0246 seconds
Elapsed time: 0.0220 seconds
Elapsed time: 0.0207 seconds
r.shape =  (31, 2000)
k = 31
relerr =  1.2354203708398645e-15
Elapsed time: 0.1908 seconds
Elapsed time: 0.1736 seconds
Elapsed time: 0.1735 seconds
Elapsed time: 0.1740 seconds
Elapsed time: 0.1739 seconds
Elapsed time: 0.1751 seconds
Elapsed time: 0.1733 seconds
Elapsed time: 0.1751 seconds
Elapsed time: 0.1737 seconds
Flam k= 30
Flam relerr =  8.195141403621299e-15



-------------------------------------------------------------------------------

Using (local) python:

(local) $ python3
Python 3.10.12 (main, Jan 17 2025, 14:35:34) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> np.show_config()
/local/tmp/zng2/repositories/libid/python/local/lib/python3.10/site-packages/numpy/__config__.py:155: UserWarning: Install `pyyaml` for better output
  warnings.warn("Install `pyyaml` for better output", stacklevel=1)
{
  "Compilers": {
    "c": {
      "name": "gcc",
      "linker": "ld.bfd",
      "version": "10.2.1",
      "commands": "cc"
    },
    "cython": {
      "name": "cython",
      "linker": "cython",
      "version": "3.1.0",
      "commands": "cython"
    },
    "c++": {
      "name": "gcc",
      "linker": "ld.bfd",
      "version": "10.2.1",
      "commands": "c++"
    }
  },
  "Machine Information": {
    "host": {
      "cpu": "x86_64",
      "family": "x86_64",
      "endian": "little",
      "system": "linux"
    },
    "build": {
      "cpu": "x86_64",
      "family": "x86_64",
      "endian": "little",
      "system": "linux"
    }
  },
 "Build Dependencies": {
    "blas": {
      "name": "scipy-openblas",
      "found": true,
      "version": "0.3.29",
      "detection method": "pkgconfig",
      "include directory": "/opt/_internal/cpython-3.10.15/lib/python3.10/site-packages/scipy_openblas64/include",
      "lib directory": "/opt/_internal/cpython-3.10.15/lib/python3.10/site-packages/scipy_openblas64/lib",
      "openblas configuration": "OpenBLAS 0.3.29  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64",
      "pc file directory": "/project/.openblas"
    },
    "lapack": {
      "name": "scipy-openblas",
      "found": true,
      "version": "0.3.29",
      "detection method": "pkgconfig",
      "include directory": "/opt/_internal/cpython-3.10.15/lib/python3.10/site-packages/scipy_openblas64/include",
      "lib directory": "/opt/_internal/cpython-3.10.15/lib/python3.10/site-packages/scipy_openblas64/lib",
      "openblas configuration": "OpenBLAS 0.3.29  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64",
      "pc file directory": "/project/.openblas"
    }
  },
  "Python Information": {
    "path": "/tmp/build-env-a8ncef9o/bin/python",
    "version": "3.10"
  },
  "SIMD Extensions": {
    "baseline": [
      "SSE",
      "SSE2",
      "SSE3"
    ],
    "found": [
      "SSSE3",
      "SSE41",
      "POPCNT",
      "SSE42",
      "AVX",
      "F16C",
      "FMA3",
      "AVX2",
      "AVX512F",
      "AVX512CD",
      "AVX512_SKX",
      "AVX512_CLX"
    ],
    "not found": [
      "AVX512_KNL",
      "AVX512_KNM",
      "AVX512_CNL",
      "AVX512_ICL"
    ]
  }
}


Using system python:

$ python3
Python 3.10.12 (main, Jan 17 2025, 14:35:34) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.

>>> import numpy as np
>>> np.show_config()
blas_mkl_info:
  NOT AVAILABLE
blis_info:
  NOT AVAILABLE
openblas_info:
  NOT AVAILABLE
accelerate_info:
  NOT AVAILABLE
atlas_3_10_blas_threads_info:
  NOT AVAILABLE
atlas_3_10_blas_info:
  NOT AVAILABLE
atlas_blas_threads_info:
  NOT AVAILABLE
atlas_blas_info:
  NOT AVAILABLE
blas_info:
    libraries = ['blas', 'blas']
    library_dirs = ['/usr/lib/x86_64-linux-gnu']
    include_dirs = ['/usr/local/include', '/usr/include']
    language = c
    define_macros = [('HAVE_CBLAS', None)]
blas_opt_info:
    define_macros = [('NO_ATLAS_INFO', 1), ('HAVE_CBLAS', None)]
    libraries = ['blas', 'blas']
    library_dirs = ['/usr/lib/x86_64-linux-gnu']
    include_dirs = ['/usr/local/include', '/usr/include']
    language = c
lapack_mkl_info:
  NOT AVAILABLE
openblas_lapack_info:
  NOT AVAILABLE
openblas_clapack_info:
  NOT AVAILABLE
flame_info:
  NOT AVAILABLE
atlas_3_10_threads_info:
  NOT AVAILABLE
atlas_3_10_info:
  NOT AVAILABLE
atlas_threads_info:
  NOT AVAILABLE
atlas_info:
  NOT AVAILABLE
lapack_info:
    libraries = ['lapack', 'lapack']
    library_dirs = ['/usr/lib/x86_64-linux-gnu']
    language = f77
lapack_opt_info:
    libraries = ['lapack', 'lapack', 'blas', 'blas']
    library_dirs = ['/usr/lib/x86_64-linux-gnu']
    language = c
    define_macros = [('NO_ATLAS_INFO', 1), ('HAVE_CBLAS', None)]
    include_dirs = ['/usr/local/include', '/usr/include']
Supported SIMD extensions in this NumPy install:
    baseline = SSE,SSE2,SSE3
    found = SSSE3,SSE41,POPCNT,SSE42,AVX,F16C,FMA3,AVX2,AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX
