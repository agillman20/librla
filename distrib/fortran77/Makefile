# librla Fortran 77 port
#
#   make test                      build and run the test suite
#   make test-quad                 the same, promoted to quad precision
#   make demo                      build and run the demo
#   make lib                       build build/librla77.a
#   make clean
#   make generate                  regenerate the real/complex halves from tools/
#
# Self-contained: no BLAS or LAPACK to link. -std=legacy because real *8
# and complex *16 are (universal) extensions to Fortran 77.
#
# Author: Adrianna Gillman, Zydrunas Gimbutas
# SPDX-License-Identifier: MIT
# Assisted by: Claude Code (Anthropic)

ifeq ($(origin FC),default)
FC = gfortran
endif

OPT ?= -O2
WARN = -std=legacy -Wall
QUAD = -freal-8-real-16 -fdefault-integer-8
BUILD ?= build
FFLAGS += $(OPT) $(WARN)

.PHONY: all lib test test-quad demo clean generate

all: lib $(BUILD)/test_all $(BUILD)/demo_basic

lib: $(BUILD)/librla77.a

test: $(BUILD)/test_all
	./$(BUILD)/test_all

test-quad: $(BUILD)-quad/test_all
	./$(BUILD)-quad/test_all

demo: $(BUILD)/demo_basic
	./$(BUILD)/demo_basic

$(BUILD) $(BUILD)-quad:
	mkdir -p $@

$(BUILD)/librla77.o: src/librla77.f | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/librla77.a: $(BUILD)/librla77.o
	rm -f $@
	ar rcs $@ $^

$(BUILD)/test_all: test/test_all.f $(BUILD)/librla77.a
	$(FC) $(FFLAGS) $< $(BUILD)/librla77.a -o $@

$(BUILD)/demo_basic: demo/demo_basic.f $(BUILD)/librla77.a
	$(FC) $(FFLAGS) $< $(BUILD)/librla77.a -o $@

$(BUILD)-quad/librla77.o: src/librla77.f | $(BUILD)-quad
	$(FC) $(FFLAGS) $(QUAD) -c $< -o $@

$(BUILD)-quad/test_all: test/test_all.f $(BUILD)-quad/librla77.o
	$(FC) $(FFLAGS) $(QUAD) $< $(BUILD)-quad/librla77.o -o $@

clean:
	rm -rf $(BUILD) $(BUILD)-quad

# src/librla77.f and test/test_all.f are generated; edit the templates in
# tools/ (and tools/test_harness.f)
generate:
	python3 tools/generate.py
