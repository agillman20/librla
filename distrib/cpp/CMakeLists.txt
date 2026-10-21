# librla C++ port: header-only library, test and demo executables
#
# Author: Adrianna Gillman, Zydrunas Gimbutas
# SPDX-License-Identifier: MIT
# Assisted by: Claude Code (Anthropic)
#
#   cmake -S . -B build [-DLIBRLA_LAPACK=Accelerate|OpenBLAS|System]
#   cmake --build build
#   ctest --test-dir build --output-on-failure

cmake_minimum_required(VERSION 3.16)
project(librla_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- LAPACK/BLAS backend ----
if(APPLE)
  set(_librla_default_lapack Accelerate)
  set(_librla_default_openblas /opt/homebrew/opt/openblas)
else()
  set(_librla_default_lapack System)
  set(_librla_default_openblas "")
endif()
set(LIBRLA_LAPACK ${_librla_default_lapack} CACHE STRING
    "LAPACK/BLAS backend: Accelerate, OpenBLAS or System")
set_property(CACHE LIBRLA_LAPACK PROPERTY STRINGS Accelerate OpenBLAS System)
set(OPENBLAS_ROOT ${_librla_default_openblas} CACHE PATH
    "OpenBLAS install prefix (LIBRLA_LAPACK=OpenBLAS)")
option(LIBRLA_ACCELERATE_LEGACY
       "Accelerate: bind the legacy LAPACK 3.2.1 symbols (name_) instead of name$NEWLAPACK (macOS >= 13.3)"
       OFF)

add_library(librla INTERFACE)
add_library(librla::librla ALIAS librla)
target_include_directories(librla INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(LIBRLA_LAPACK STREQUAL "Accelerate")
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  target_link_libraries(librla INTERFACE ${ACCELERATE_FRAMEWORK})
  if(NOT LIBRLA_ACCELERATE_LEGACY)
    target_compile_definitions(librla INTERFACE LIBRLA_ACCELERATE_NEWLAPACK)
    set(_librla_iface " (new LAPACK interface, name$NEWLAPACK)")
  else()
    set(_librla_iface " (legacy interface, name_)")
  endif()
elseif(LIBRLA_LAPACK STREQUAL "OpenBLAS")
  find_library(OPENBLAS_LIBRARY NAMES openblas
               HINTS ${OPENBLAS_ROOT}/lib ${OPENBLAS_ROOT}/lib64 REQUIRED)
  target_link_libraries(librla INTERFACE ${OPENBLAS_LIBRARY})
elseif(LIBRLA_LAPACK STREQUAL "System")
  find_package(LAPACK REQUIRED)
  target_link_libraries(librla INTERFACE LAPACK::LAPACK)
else()
  message(FATAL_ERROR "LIBRLA_LAPACK must be Accelerate, OpenBLAS or System")
endif()
message(STATUS "librla: LAPACK backend = ${LIBRLA_LAPACK}${_librla_iface}")

# ---- executables ----
set(LIBRLA_WARNINGS -Wall -Wextra -Wpedantic)

add_executable(test_all test/test_all.cpp)
target_link_libraries(test_all PRIVATE librla)
target_compile_options(test_all PRIVATE ${LIBRLA_WARNINGS})

add_executable(demo_basic demo/demo_basic.cpp)
target_link_libraries(demo_basic PRIVATE librla)
target_compile_options(demo_basic PRIVATE ${LIBRLA_WARNINGS})

enable_testing()
add_test(NAME test_all COMMAND test_all)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
                  DEPENDS test_all WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
