# librla Fortran port
#
#   make test                      build and run the test suite
#   make demo                      build and run the demo
#   make lib                       build build/librla.a and the .mod files
#   make clean
#
# LAPACK/BLAS backend (default: Accelerate on macOS, -llapack -lblas elsewhere):
#   make test LAPACK="-L/opt/homebrew/opt/openblas/lib -lopenblas"
# With Accelerate, librla binds the LAPACK 3.11 interface (dgemm$NEWLAPACK ...,
# macOS 13.3+); on older macOS use the legacy symbols:
#   make test NEWLAPACK=0
# Changing LAPACK or NEWLAPACK rebuilds what depends on it automatically.
#
# Author: Adrianna Gillman, Zydrunas Gimbutas
# SPDX-License-Identifier: MIT
# Assisted by: Claude Code (Anthropic)

ifeq ($(origin FC),default)
FC = gfortran
endif

ifeq ($(shell uname -s),Darwin)
LAPACK ?= -framework Accelerate
else
LAPACK ?= -llapack -lblas
endif

# Accelerate's new LAPACK symbols: only when linking Accelerate
NEWLAPACK ?= 1
ifneq ($(findstring Accelerate,$(LAPACK)),)
ifeq ($(NEWLAPACK),1)
LAPACK_DEFS = -DLIBRLA_NEWLAPACK
endif
endif

OPT ?= -O2
WARN = -std=f2003 -Wall -Wextra
BUILD ?= build
FFLAGS += $(OPT) $(WARN) -J$(BUILD) -I$(BUILD)

LIB_OBJ = $(BUILD)/librla_lapack.o $(BUILD)/librla.o
LIBRLA = $(BUILD)/librla.a

.PHONY: all lib test demo clean FORCE

all: lib $(BUILD)/test_all $(BUILD)/demo_basic

lib: $(LIBRLA)

test: $(BUILD)/test_all
	./$(BUILD)/test_all

demo: $(BUILD)/demo_basic
	./$(BUILD)/demo_basic

$(BUILD):
	mkdir -p $(BUILD)

# Stamps recording the backend so that switching it rebuilds the LAPACK
# binding (lapack.defs) and relinks the executables (lapack.flags).
$(BUILD)/lapack.flags: FORCE | $(BUILD)
	@echo '$(LAPACK)' | cmp -s - $@ || echo '$(LAPACK)' > $@

$(BUILD)/lapack.defs: FORCE | $(BUILD)
	@echo '$(LAPACK_DEFS)' | cmp -s - $@ || echo '$(LAPACK_DEFS)' > $@

$(BUILD)/librla_lapack.o: src/librla_lapack.F90 $(BUILD)/lapack.defs | $(BUILD)
	$(FC) $(FFLAGS) $(LAPACK_DEFS) -c $< -o $@

$(BUILD)/librla.o: src/librla.f90 $(BUILD)/librla_lapack.o | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@

$(LIBRLA): $(LIB_OBJ)
	rm -f $@
	ar rcs $@ $^

$(BUILD)/test_all: test/test_all.f90 $(LIBRLA) $(BUILD)/lapack.flags
	$(FC) $(FFLAGS) $< $(LIBRLA) $(LAPACK) -o $@

$(BUILD)/demo_basic: demo/demo_basic.f90 $(LIBRLA) $(BUILD)/lapack.flags
	$(FC) $(FFLAGS) $< $(LIBRLA) $(LAPACK) -o $@

clean:
	rm -rf $(BUILD)

FORCE:
