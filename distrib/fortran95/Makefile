# librla Fortran 95 port
#
#   make test                      build and run the test suite
#   make demo                      build and run the demo
#   make lib                       build build/librla95.a (and librla95.mod)
#   make clean
#   make generate                  regenerate the real/complex halves from tools/
#
# LAPACK/BLAS backend (default: Accelerate on macOS, -llapack -lblas elsewhere):
#   make test LAPACK="-L/opt/homebrew/opt/openblas/lib -lopenblas"
# Changing LAPACK relinks the executables automatically.
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

OPT ?= -O2
WARN = -std=f95 -Wall -Wextra
BUILD ?= build
FFLAGS += $(OPT) $(WARN) -J$(BUILD) -I$(BUILD)

LIB_OBJ = $(BUILD)/rla95_lapack.o $(BUILD)/rla95_core.o $(BUILD)/librla95.o \
          $(BUILD)/librla95_interfaces.o
LIBRLA = $(BUILD)/librla95.a

.PHONY: all lib test demo clean generate FORCE

all: lib $(BUILD)/test_all $(BUILD)/demo_basic

lib: $(LIBRLA)

test: $(BUILD)/test_all
	./$(BUILD)/test_all

demo: $(BUILD)/demo_basic
	./$(BUILD)/demo_basic

$(BUILD):
	mkdir -p $(BUILD)

# Stamp recording the backend so that switching it relinks the executables
$(BUILD)/lapack.flags: FORCE | $(BUILD)
	@echo '$(LAPACK)' | cmp -s - $@ || echo '$(LAPACK)' > $@

$(BUILD)/rla95_lapack.o: src/rla95_lapack.f90 | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/rla95_core.o: src/rla95_core.f90 $(BUILD)/rla95_lapack.o | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/librla95.o: src/librla95.f90 $(BUILD)/rla95_core.o | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/librla95_interfaces.o: src/librla95_interfaces.f90 | $(BUILD)
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

# src/rla95_core.f90, src/librla95.f90, src/librla95_interfaces.f90 and
# test/test_all.f90 are generated; edit the templates in tools/
generate:
	python3 tools/generate.py

FORCE:
