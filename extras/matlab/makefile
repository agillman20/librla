all: linux clean

linux: linux-octave-64bit

macos: macos-octave-64bit

windows: windows-matlab-64bit

linux-octave-64bit:
	$(MAKE) -f makefile.mwrap TARGET=octave-linux-openmp clean
	$(MAKE) -f makefile.mwrap TARGET=octave-linux-openmp -j

linux-matlab-64bit:
	$(MAKE) -f makefile.mwrap TARGET=matlab-linux-a64-openmp clean
	$(MAKE) -f makefile.mwrap TARGET=matlab-linux-a64-openmp -j

macos-octave-64bit:
	$(MAKE) -f makefile.mwrap TARGET=octave-macos-openmp clean
	$(MAKE) -f makefile.mwrap TARGET=octave-macos-openmp -j

macos-matlab-64bit:
	$(MAKE) -f makefile.mwrap TARGET=matlab-maca64-openmp clean
	$(MAKE) -f makefile.mwrap TARGET=matlab-maca64-openmp -j

windows-matlab-64bit:
	$(MAKE) -f makefile.mwrap TARGET=matlab-windows-w64-openmp clean 
	$(MAKE) -f makefile.mwrap TARGET=matlab-windows-w64-openmp -j

clean:
	rm -f *.o

distclean:
	rm -f *.o

