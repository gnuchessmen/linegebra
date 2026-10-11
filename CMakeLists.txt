# Top-level build description for linegebra.
#
# Typical use:
#   cmake -S . -B build        # configure: read this file, generate build files in build/
#   cmake --build build        # compile the tests and examples
#   ctest --test-dir build     # run the tests

cmake_minimum_required(VERSION 3.21)

project(linegebra
    VERSION 0.0.1
    DESCRIPTION "A header-only dense linear algebra library"
    LANGUAGES CXX)

# ---------------------------------------------------------------------------
# The library itself
# ---------------------------------------------------------------------------
# linegebra is header-only: there is nothing to compile, so it is an
# INTERFACE library. Linking to it only adds the include directory and the
# C++20 requirement to whoever uses it.
add_library(linegebra INTERFACE)
add_library(linegebra::linegebra ALIAS linegebra)

target_include_directories(linegebra INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_compile_features(linegebra INTERFACE cxx_std_20)

# ---------------------------------------------------------------------------
# Tests and examples (only built when linegebra is the main project, not when
# someone else pulls it into their own build)
# ---------------------------------------------------------------------------
option(LINEGEBRA_BUILD_TESTS "Build the linegebra tests" ${PROJECT_IS_TOP_LEVEL})
option(LINEGEBRA_BUILD_EXAMPLES "Build the linegebra examples" ${PROJECT_IS_TOP_LEVEL})

# Strict warnings for our own code. Applied to tests and examples, not
# forced on users of the library.
add_library(linegebra_warnings INTERFACE)
if(MSVC)
    target_compile_options(linegebra_warnings INTERFACE /W4 /WX /permissive-)
else()
    target_compile_options(linegebra_warnings INTERFACE -Wall -Wextra -Wpedantic -Werror)
endif()

if(LINEGEBRA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(LINEGEBRA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
