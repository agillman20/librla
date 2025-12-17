#
#  Generate Matlab/Octave MEX files
#  Date: Mar 25, 2025
#


###TARGET=octave-linux
###TARGET=octave-linux-openmp

###TARGET=matlab-linux-a64
###TARGET=matlab-linux-a64-openmp

###TARGET=octave-macos
###TARGET=octave-macos-openmp

###TARGET=matlab-maci64
###TARGET=matlab-maci64-openmp

###TARGET=matlab-maca64
###TARGET=matlab-maca64-openmp

###TARGET=matlab-windows-w64
###TARGET=matlab-windows-w64-openmp



##############################################################################
#
#  Octave | Linux 64-bit | gfortran compiler
#
##############################################################################


ifeq ($(TARGET), octave-linux)

FFLAGS=-g -O3 -std=legacy -fPIC

# Use the following for GNU Octave + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    gfortran
CC=    gcc
CXX=   g++
MEX=   FFLAGS="-03 -std=legacy -fPIC" mkoctfile --mex
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-c99complex -mex	
###MEX_LIBS=-lm
###MEX_LIBS=-Wl,--no-undefined -lm -ldl -lgfortran -loctinterp
MEX_LIBS=-lm -ldl -lgfortran -loctinterp
MEX_LINK=

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


ifeq ($(TARGET), octave-linux-openmp)

FFLAGS=-g -O3 -std=legacy --openmp -fPIC

# Use the following for GNU Octave + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    gfortran
CC=    gcc
CXX=   g++
MEX=   FFLAGS="-g -O3 -std=legacy --openmp -fPIC" mkoctfile --mex 
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-c99complex -mex	
###MEX_LIBS=-lgomp -lm
###MEX_LIBS=-Wl,--no-undefined -lgomp -lm -ldl -lgfortran -loctinterp
MEX_LIBS=-lgomp -lm -ldl -lgfortran -loctinterp
MEX_LINK=

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


ifeq ($(TARGET), octave-linux-ilp64)

FFLAGS=-g -O3 -std=legacy -fPIC -finteger-4-integer-8

# Use the following for GNU Octave + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    gfortran
CC=    gcc
CXX=   g++
MEX=   FFLAGS="-03 -std=legacy -fPIC -finteger-4-integer-8" mkoctfile --mex
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-i8 -c99complex -mex	
###MEX_LIBS=-lm
###MEX_LIBS=-Wl,--no-undefined -lm -ldl -lgfortran -loctinterp
MEX_LIBS=-lm -ldl -lgfortran -loctinterp
MEX_LINK=

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


ifeq ($(TARGET), octave-linux-ilp64-openmp)

FFLAGS=-g -O3 -std=legacy --openmp -fPIC -finteger-4-integer-8

# Use the following for GNU Octave + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    gfortran
CC=    gcc
CXX=   g++
MEX=   FFLAGS="-g -O3 -std=legacy --openmp -fPIC -finteger-4-integer-8" mkoctfile --mex 
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-i8 -c99complex -mex	
###MEX_LIBS=-lgomp -lm
###MEX_LIBS=-Wl,--no-undefined -lgomp -lm -ldl -lgfortran -loctinterp
MEX_LIBS=-lgomp -lm -ldl -lgfortran -loctinterp
MEX_LINK=

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


##############################################################################
#
#  Octave | Windows 64-bit | gfortran compiler
#
##############################################################################



ifeq ($(TARGET), octave-windows)

FFLAGS=-g -O3 -std=legacy -fPIC

# Use the following for GNU Octave + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    gfortran
CC=    gcc
CXX=   g++
MEX=   FFLAGS="-03 -std=legacy -fPIC" mkoctfile --mex
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-c99complex -mex	
###MEX_LIBS=-lm
###MEX_LIBS=-Wl,--no-undefined -lm -ldl -lgfortran -loctinterp
MEX_LIBS=-lm -lgfortran -loctinterp
MEX_LINK=

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


ifeq ($(TARGET), octave-windows-openmp)

FFLAGS=-g -O3 -std=legacy --openmp -fPIC

# Use the following for GNU Octave + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    gfortran
CC=    gcc
CXX=   g++
MEX=   FFLAGS="-g -O3 -std=legacy --openmp -fPIC" mkoctfile --mex 
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-c99complex -mex	
###MEX_LIBS=-lgomp -lm
###MEX_LIBS=-Wl,--no-undefined -lgomp -lm -ldl -lgfortran -loctinterp
MEX_LIBS=-lgomp -lm -lgfortran -loctinterp
MEX_LINK=

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


ifeq ($(TARGET), octave-windows-ilp64)

FFLAGS=-g -O3 -std=legacy -fPIC -finteger-4-integer-8

# Use the following for GNU Octave + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    gfortran
CC=    gcc
CXX=   g++
MEX=   FFLAGS="-03 -std=legacy -fPIC -finteger-4-integer-8" mkoctfile --mex
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-i8 -c99complex -mex	
###MEX_LIBS=-lm
###MEX_LIBS=-Wl,--no-undefined -lm -lgfortran -loctinterp
MEX_LIBS=-lm -lgfortran -loctinterp
MEX_LINK=

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


ifeq ($(TARGET), octave-windows-ilp64-openmp)

FFLAGS=-g -O3 -std=legacy --openmp -fPIC -finteger-4-integer-8

# Use the following for GNU Octave + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    gfortran
CC=    gcc
CXX=   g++
MEX=   FFLAGS="-g -O3 -std=legacy --openmp -fPIC -finteger-4-integer-8" mkoctfile --mex 
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-i8 -c99complex -mex	
###MEX_LIBS=-lgomp -lm
###MEX_LIBS=-Wl,--no-undefined -lgomp -lm -lgfortran -loctinterp
MEX_LIBS=-lgomp -lm -lgfortran -loctinterp
MEX_LINK=

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


##############################################################################
#
#  Octave | MacOS 64-bit | gfortran compiler
#
##############################################################################


ifeq ($(TARGET), octave-macos)

FFLAGS=-g -O3 -std=legacy -fPIC

# Use the following for GNU Octave + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    gfortran
CC=    gcc
CXX=   g++
MEX=   FFLAGS="-g -O3 -std=legacy -fPIC" mkoctfile --mex
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-c99complex -mex    
###MEX_LIBS=-lf2c -lm
FORTRAN_LIBRARY_PATH=$(dir $(shell gfortran --print-file-name libgfortran.dylib))
MEX_LIBS=-L$(FORTRAN_LIBRARY_PATH) -Wl,-rpath,$(FORTRAN_LIBRARY_PATH) -Wl,-no_compact_unwind -L. -lgfortran -lm -ldl
###MEX_LIBS=-lgfortran -lm -ldl
MEX_LINK=

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


ifeq ($(TARGET), octave-macos-openmp)

FFLAGS=-g -O3 -std=legacy --openmp -fPIC

# Use the following for GNU Octave + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    gfortran
CC=    gcc
CXX=   g++
MEX=   FFLAGS="-g -O3 -std=legacy --openmp -fPIC" mkoctfile --mex 
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-c99complex -mex
###MEX_LIBS=-lgomp -lf2c -lm
FORTRAN_LIBRARY_PATH=$(dir $(shell gfortran --print-file-name libgfortran.dylib))
MEX_LIBS=-L$(FORTRAN_LIBRARY_PATH) -Wl,-rpath,$(FORTRAN_LIBRARY_PATH) -Wl,-no_compact_unwind -L. -lgfortran -lgomp -lm -ldl
###MEX_LIBS=-lgfortran -gomp -lm -ldl
MEX_LINK=

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif



ifeq ($(TARGET), octave-macos-ilp64)

FFLAGS=-g -O3 -std=legacy -fPIC -finteger-4-integer-8

# Use the following for GNU Octave + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    gfortran
CC=    gcc
CXX=   g++
MEX=   FFLAGS="-g -O3 -std=legacy -fPIC -finteger-4-integer-8" mkoctfile --mex
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-i8 -c99complex -mex
###MEX_LIBS=-lf2c -lm
FORTRAN_LIBRARY_PATH=$(dir $(shell gfortran --print-file-name libgfortran.dylib))
MEX_LIBS=-L$(FORTRAN_LIBRARY_PATH) -Wl,-rpath,$(FORTRAN_LIBRARY_PATH) -Wl,-no_compact_unwind -L. -lgfortran -lm -ldl
###MEX_LIBS=-lgfortran -lm -ldl
MEX_LINK=

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


ifeq ($(TARGET), octave-macos-ilp64-openmp)

FFLAGS=-g -O3 -std=legacy --openmp -fPIC -finteger-4-integer-8

# Use the following for GNU Octave + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    gfortran
CC=    gcc
CXX=   g++
MEX=   FFLAGS="-g -O3 -std=legacy --openmp -fPIC -finteger-4-integer-8" mkoctfile --mex 
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-i8 -c99complex -mex
###MEX_LIBS=-lgomp -lf2c -lm
FORTRAN_LIBRARY_PATH=$(dir $(shell gfortran --print-file-name libgfortran.dylib))
MEX_LIBS=-L$(FORTRAN_LIBRARY_PATH) -Wl,-rpath,$(FORTRAN_LIBRARY_PATH) -Wl,-no_compact_unwind -L. -lgfortran -lgomp -lm -ldl
###MEX_LIBS=-lgfortran -gomp -lm -ldl
MEX_LINK=

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


##############################################################################
#
#  Matlab | MacOS 64-bit | Intel | gfortran compiler
#
##############################################################################


ifeq ($(TARGET), matlab-maci64)

FFLAGS=-g -O3 -std=legacy --openmp -fPIC

# Use the following for Matlab + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    gfortran
CC=    gcc
CXX=   g++
# Uncomment this line for new-style classdef support
# OOFLAG=-DR2008OO
MEX=   mex $(OOFLAG) CC=gcc LD=gcc ARCH=maci64
# Use the following for 64-bit MEX
###MEX=   mex -largeArrayDims $(OOFLAG) ARCH=maci64
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-c99complex -mex        
FORTRAN_LIBRARY_PATH=$(dir $(shell gfortran --print-file-name libgfortran.dylib))
MEX_LIBS=-L$(FORTRAN_LIBRARY_PATH) -L. -lgfortran -lm -ldl
###MEX_LIBS=-lgfortran -lm
MEX_LINK=

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


ifeq ($(TARGET), matlab-maci64-openmp)

FFLAGS=-g -O3 -std=legacy -fPIC --openmp

# Use the following for Matlab + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    gfortran
CC=    gcc
CXX=   g++
# Uncomment this line for new-style classdef support
# OOFLAG=-DR2008OO
MEX=   mex $(OOFLAG) CC=gcc LD=gcc ARCH=maci64
# Use the following for 64-bit MEX
###MEX=   mex -largeArrayDims $(OOFLAG) ARCH=maci64
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-c99complex -mex        
FORTRAN_LIBRARY_PATH=$(dir $(shell gfortran --print-file-name libgfortran.dylib))
MEX_LIBS=-L$(FORTRAN_LIBRARY_PATH) -L. -lgfortran -lgomp -lm
###MEX_LIBS=-lgfortran -lgomp -lm
MEX_LINK=

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


##############################################################################
#
#  Matlab | MacOS 64-bit | Apple Silicon | gfortran compiler
#
##############################################################################


ifeq ($(TARGET), matlab-maca64)

FFLAGS=-g -O3 -std=legacy --openmp -fPIC

# Use the following for Matlab + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    gfortran
CC=    gcc
CXX=   g++
# Uncomment this line for new-style classdef support
# OOFLAG=-DR2008OO
###MEX=   mex $(OOFLAG) CC=gcc LD=gcc ARCH=maca64
# Use the following for 64-bit MEX
MEX=   mex -largeArrayDims $(OOFLAG) ARCH=maca64
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-c99complex -mex        
FORTRAN_LIBRARY_PATH=$(dir $(shell gfortran --print-file-name libgfortran.dylib))
MEX_LIBS=-L$(FORTRAN_LIBRARY_PATH) -L. -lgfortran -lm -ldl
###MEX_LIBS=-lgfortran -lm
MEX_LINK=

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif



ifeq ($(TARGET), matlab-maca64-openmp)

FFLAGS=-g -O3 -std=legacy -fPIC --openmp

# Use the following for Matlab + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    gfortran
CC=    gcc
CXX=   g++
# Uncomment this line for new-style classdef support
# OOFLAG=-DR2008OO
###MEX=   mex $(OOFLAG) CC=gcc LD=gcc ARCH=maca64
# Use the following for 64-bit MEX
MEX=   mex -largeArrayDims $(OOFLAG) ARCH=maca64
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-c99complex -mex        
FORTRAN_LIBRARY_PATH=$(dir $(shell gfortran --print-file-name libgfortran.dylib))
MEX_LIBS=-L$(FORTRAN_LIBRARY_PATH) -L. -lgfortran -lgomp -lm
###MEX_LIBS=-lgfortran -lgomp -lm
MEX_LINK=

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


##############################################################################
#
#  Matlab | Linux 64-bit | gfortran compiler
#
##############################################################################


ifeq ($(TARGET), matlab-linux-a64)

FFLAGS=-g -O3 -std=legacy -fPIC

# Use the following for Matlab + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    gfortran
CC=    gcc
CXX=   g++
# Uncomment this line for new-style classdef support
# OOFLAG=-DR2008OO
###MEX=   mex $(OOFLAG)
# Use the following for 64-bit MEX
MEX=   mex -largeArrayDims $(OOFLAG)
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-c99complex -mex	
MEX_LIBS=-lgfortran -lm
MEX_LINK=-output $(MEX_GATEWAY).mexa64

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif



ifeq ($(TARGET), matlab-linux-a64-openmp)

FFLAGS=-g -O3 -std=legacy --openmp -fPIC

# Use the following for Matlab + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    gfortran
CC=    gcc
CXX=   g++
# Uncomment this line for new-style classdef support
# OOFLAG=-DR2008OO
###MEX=   mex $(OOFLAG)
# Use the following for 64-bit MEX
MEX=   mex -largeArrayDims $(OOFLAG)
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-c99complex -mex	
MEX_LIBS=-lgomp -lgfortran -lm
MEX_LINK=-output $(MEX_GATEWAY).mexa64

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


ifeq ($(TARGET), matlab-linux-ilp64-a64)

FFLAGS=-g -O3 -std=legacy -fPIC -finteger-4-integer-8

# Use the following for Matlab + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    gfortran
CC=    gcc
CXX=   g++
# Uncomment this line for new-style classdef support
# OOFLAG=-DR2008OO
###MEX=   mex $(OOFLAG)
# Use the following for 64-bit MEX
MEX=   mex -largeArrayDims $(OOFLAG)
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-i8 -c99complex -mex	
MEX_LIBS=-lgfortran -lm
MEX_LINK=-output $(MEX_GATEWAY).mexa64

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif



ifeq ($(TARGET), matlab-linux-ilp64-a64-openmp)

FFLAGS=-g -O3 -std=legacy --openmp -fPIC -finteger-4-integer-8

# Use the following for Matlab + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    gfortran
CC=    gcc
CXX=   g++
# Uncomment this line for new-style classdef support
# OOFLAG=-DR2008OO
###MEX=   mex $(OOFLAG)
# Use the following for 64-bit MEX
MEX=   mex -largeArrayDims $(OOFLAG)
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-i8 -c99complex -mex	
MEX_LIBS=-lgomp -lgfortran -lm
MEX_LINK=-output $(MEX_GATEWAY).mexa64

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


##############################################################################
#
#  Matlab | Windows 64-bit | MinGW cross_win64 compiler under Linux
#
##############################################################################


ifeq ($(TARGET), matlab-windows-w64)

# Use the following for Matlab + gfortran cross compiler.
TESTC99COMPLEX=test_c99_complex

CROSS_FC=gfortran -g -O3 -shared -fexceptions -I/C/Program\ Files/MATLAB/R2023a/extern/include
CROSS_CC=gcc -g -O3 -shared -fexceptions -I/C/Program\ Files/MATLAB/R2023a/extern/include
CROSS_LIBS=-I/C/Program\ Files/MATLAB/R2023a/extern/include -L/C/Program\ Files/MATLAB/R2023a/bin/win64 -lmex -lmat -lmx -lm -lgfortran -lquadmath

FC=    $(CROSS_FC) --std=legacy
CC=    $(CROSS_CC) --std=legacy
CXX=   $(CROSS_CC) --std=legacy
MEX=   $(CXX)
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-c99complex -mex	
MEX_LIBS=$(CROSS_LIBS) -Wl,--no-undefined 
MEX_LINK=-o $(MEX_GATEWAY).mexw64

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


ifeq ($(TARGET), matlab-windows-w64-openmp)

# Use the following for Matlab + gfortran cross compiler.
TESTC99COMPLEX=test_c99_complex

CROSS_FC=gfortran -g -O3 -shared -fexceptions -I/C/Program\ Files/MATLAB/R2023a/extern/include
CROSS_CC=gcc -g -O3 -shared -fexceptions -I/C/Program\ Files/MATLAB/R2023a/extern/include
CROSS_LIBS=-I/C/Program\ Files/MATLAB/R2023a/extern/include -L/C/Program\ Files/MATLAB/R2023a/bin/win64 -lmex -lmat -lmx -lm -lgfortran -lquadmath

FC=    $(CROSS_FC)  --openmp --std=legacy
CC=    $(CROSS_CC)  --openmp --std=legacy
CXX=   $(CROSS_CC)  --openmp --std=legacy
MEX=   $(CXX)
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-c99complex -mex
MEX_LIBS=$(CROSS_LIBS) --openmp -lgomp -lpthread -Wl,--no-undefined 
MEX_LINK=-o $(MEX_GATEWAY).mexw64

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif



ifeq ($(TARGET), matlab-windows-ilp64-w64)

# Use the following for Matlab + gfortran cross compiler.
TESTC99COMPLEX=test_c99_complex

CROSS_FC=gfortran -g -O3 -shared -fexceptions -I/C/Program\ Files/MATLAB/R2023a/extern/include
CROSS_CC=gcc -g -O3 -shared -fexceptions -I/C/Program\ Files/MATLAB/R2023a/extern/include
CROSS_LIBS=-I/C/Program\ Files/MATLAB/R2023a/extern/include -L/C/Program\ Files/MATLAB/R2023a/bin/win64 -lmex -lmat -lmx -lm -lgfortran -lquadmath

FC=    $(CROSS_FC) --std=legacy -finteger-4-integer-8 
CC=    $(CROSS_CC) --std=legacy -finteger-4-integer-8 
CXX=   $(CROSS_CC) --std=legacy -finteger-4-integer-8 
MEX=   $(CXX)
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-i8 -c99complex -mex	
MEX_LIBS=$(CROSS_LIBS) -Wl,--no-undefined 
MEX_LINK=-o $(MEX_GATEWAY).mexw64

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


ifeq ($(TARGET), matlab-windows-ilp64-w64-openmp)

# Use the following for Matlab + gfortran cross compiler.
TESTC99COMPLEX=test_c99_complex

CROSS_FC=gfortran -g -O3 -shared -fexceptions -I/C/Program\ Files/MATLAB/R2023a/extern/include
CROSS_CC=gcc -g -O3 -shared -fexceptions -I/C/Program\ Files/MATLAB/R2023a/extern/include
CROSS_LIBS=-I/C/Program\ Files/MATLAB/R2023a/extern/include -L/C/Program\ Files/MATLAB/R2023a/bin/win64 -lmex -lmat -lmx -lm -lgfortran -lquadmath

FC=    $(CROSS_FC)  --openmp --std=legacy -finteger-4-integer-8 
CC=    $(CROSS_CC)  --openmp --std=legacy -finteger-4-integer-8 
CXX=   $(CROSS_CC)  --openmp --std=legacy -finteger-4-integer-8 
MEX=   $(CXX)
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-i8 -c99complex -mex
MEX_LIBS=$(CROSS_LIBS) --openmp -lgomp -lpthread -Wl,--no-undefined 
MEX_LINK=-o $(MEX_GATEWAY).mexw64

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif



##############################################################################
#
#  Matlab | Windows 64-bit | MinGW cross_win64 compiler under Linux
#
##############################################################################


ifeq ($(TARGET), matlab-windows-w64-cross)

# Use the following for Matlab + gfortran cross compiler.
TESTC99COMPLEX=test_c99_complex

CROSS_FC=$(HOME)/lib/cross_win64/bin/x86_64-w64-mingw32-gfortran -g -O3 -ffast-math -msse2 -shared -fexceptions
CROSS_CC=$(HOME)/lib/cross_win64/bin/x86_64-w64-mingw32-gcc -g -O3 -ffast-math -msse2 -shared -fexceptions
CROSS_LIBS=-I$(HOME)/lib/matlab-r2010b/extern/include -L$(HOME)/lib/matlab-r2010b/bin/win64 -lmex -lmat -lmx -lm -lgfortran -static

FC=    $(CROSS_FC)
CC=    $(CROSS_CC)
CXX=   $(CROSS_CC)
MEX=   $(CXX)
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-c99complex -mex	
MEX_LIBS=$(CROSS_LIBS) -Wl,--no-undefined 
MEX_LINK=-o $(MEX_GATEWAY).mexw64

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


ifeq ($(TARGET), matlab-windows-w64-cross-openmp)

# Use the following for Matlab + gfortran cross compiler.
TESTC99COMPLEX=test_c99_complex

CROSS_FC=$(HOME)/lib/cross_win64/bin/x86_64-w64-mingw32-gfortran -g -O3 -shared -fexceptions
CROSS_CC=$(HOME)/lib/cross_win64/bin/x86_64-w64-mingw32-gcc -g -O3 -shared -fexceptions
CROSS_LIBS=-I$(HOME)/lib/matlab-r2010b/extern/include -L$(HOME)/lib/matlab-r2010b/bin/win64 -lmex -lmat -lmx -lm -lgfortran -static

FC=    $(CROSS_FC)  --openmp
CC=    $(CROSS_CC)  --openmp
CXX=   $(CROSS_CC)  --openmp
MEX=   $(CXX)
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-c99complex -mex
MEX_LIBS=$(CROSS_LIBS) --openmp -lgomp -lpthread -Wl,--no-undefined 
MEX_LINK=-o $(MEX_GATEWAY).mexw64

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif



##############################################################################
#
#  Octave | Linux 64-bit | Intel compiler
#
##############################################################################


ifeq ($(TARGET), octave-linux-intel)

FFLAGS=-g -xT -O3 -ip -fPIC
intel_prefix="/opt/intel/fce/10.1.008/lib"

# Use the following for GNU Octave + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    ief77
CC=    gcc
CXX=   g++
MEX=   FFLAGS="-g -xT -O3 -ip -fPIC" mkoctfile --mex 
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-c99complex -mex	
###MEX_LIBS=-lgomp -lf2c -lm
MEX_LIBS=-Wl,--no-undefined -lm -ldl -Wl,--rpath,$(intel_libs)  -L$(intel_libs)  -lifcore -lifcoremt -lsvml -lirc -limf -liomp5 -ldl
MEX_LINK=

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


ifeq ($(TARGET), octave-linux-intel-openmp)

FFLAGS=-g -xT -O3 -ip -openmp -fPIC
intel_prefix="/opt/intel/fce/10.1.008/lib"

# Use the following for GNU Octave + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    ief77
CC=    gcc
CXX=   g++
MEX=   FFLAGS="-g -xT -O3 -ip -openmp -fPIC" mkoctfile --mex 
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-c99complex -mex	
###MEX_LIBS=-lgomp -lf2c -lm
MEX_LIBS=-Wl,--no-undefined -lm -ldl -Wl,--rpath,$(intel_libs)  -L$(intel_libs)  -lifcore -lifcoremt -lsvml -lirc -limf -liomp5 -ldl
MEX_LINK=

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


##############################################################################
#
#  Matlab | Linux 64-bit | Intel compiler
#
##############################################################################


ifeq ($(TARGET), matlab-linux-a64-intel)

FFLAGS=-g -xT -O3 -ip -fPIC
intel_prefix="/opt/intel/fce/10.1.008/lib"

# Use the following for Matlab + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    ief77
CC=    gcc
CXX=   g++
# Uncomment this line for new-style classdef support
# OOFLAG=-DR2008OO
###MEX=   mex $(OOFLAG)
# Use the following for 64-bit MEX
MEX=   mex -largeArrayDims $(OOFLAG)
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-c99complex -mex	
MEX_LIBS=-Wl,--rpath,$(intel_libs) -L$(intel_libs)  -lifcore -lsvml -limf -lintlc -lm
MEX_LINK=-output $(MEX_GATEWAY).mexa64

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif



ifeq ($(TARGET), matlab-linux-a64-intel-openmp)

FFLAGS=-g -xT -O3 -ip --openmp -fPIC
intel_prefix="/opt/intel/fce/10.1.008/lib"

# Use the following for Matlab + gfortran
TESTC99COMPLEX=test_c99_complex
FC=    ief77
CC=    gcc
CXX=   g++
# Uncomment this line for new-style classdef support
# OOFLAG=-DR2008OO
###MEX=   mex $(OOFLAG)
# Use the following for 64-bit MEX
MEX=   mex -largeArrayDims $(OOFLAG)
MEX_FC=   $(FC)
MEX_CC=   $(CXX)
MW=mwrap
MWFLAGS=-c99complex -mex	
MEX_LIBS=-Wl,--rpath,$(intel_libs) -L$(intel_libs) -lifcoremt -lsvml -limf -liomp5 -lintlc -lm
MEX_LINK=-output $(MEX_GATEWAY).mexa64

MEX_FLAGS=
OBJSUF=o
MODSUF=mod

endif


