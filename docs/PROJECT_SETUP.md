# How the linegebra project is set up

This guide explains every file in the project skeleton, why it exists, and
how to build and test the library on your own computer.

## The big picture

```
linegebra/
├── CMakeLists.txt              ← build instructions for the whole project
├── include/linegebra/          ← THE LIBRARY: header files users #include
│   ├── linegebra.hpp           ← "umbrella" header that includes everything
│   └── version.hpp             ← version numbers (a placeholder for now)
├── tests/                      ← automated tests (not part of the library)
│   ├── CMakeLists.txt
│   └── test_version.cpp
├── examples/                   ← small programs that use the library
│   ├── CMakeLists.txt
│   └── hello_linegebra.cpp
├── legacy/                     ← the 2021 code, kept for reference only
├── docs/                       ← documentation (this file)
├── .github/workflows/ci.yml    ← GitHub runs the tests automatically
├── .clang-format               ← code formatting rules
├── .gitignore                  ← files git should ignore (build output)
└── ROADMAP.md                  ← the plan
```

The key idea: **the library is only what is in `include/linegebra/`.**
Everything else is there to build, test, demonstrate or document it.

## Why "header-only"?

A C++ library can be shipped in two ways:

1. **Compiled** (`.h` + `.cpp` files): the `.cpp` files are compiled into a
   `.a`/`.lib` file, and users link against it. Your 2021 code was meant to
   be this kind.
2. **Header-only** (just `.hpp` files): all the code lives in headers. Users
   simply `#include` them; there is nothing to compile separately.

linegebra will be built on **templates** (`Matrix<T>` working for `float`,
`double`, complex...). Template code must be visible to the compiler
wherever it is used, so it has to live in headers anyway. Header-only is the
natural choice, and it is also the easiest for others to use.

## What is CMake?

Compiling by hand looks like:

```
g++ -std=c++20 -Iinclude tests/test_version.cpp -o tests ...
```

This quickly gets out of hand: many files, different compilers (GCC, Clang,
MSVC on Windows) with different flags, external dependencies. **CMake** is a
tool where you *describe* your project once (in `CMakeLists.txt` files), and
it generates the actual build commands for whatever compiler and system you
are on.

### Reading the top-level `CMakeLists.txt`

- `project(linegebra VERSION 0.0.1 LANGUAGES CXX)` names the project.
- `add_library(linegebra INTERFACE)` declares the library. `INTERFACE`
  means "nothing to compile, just headers".
- `target_include_directories(...)` says "anyone using linegebra should be
  able to `#include` files from `include/`". That is why you write
  `#include <linegebra/linegebra.hpp>`.
- `target_compile_features(linegebra INTERFACE cxx_std_20)` says "anyone
  using linegebra needs C++20".
- `linegebra_warnings` turns on strict compiler warnings for *our* tests and
  examples and treats warnings as errors, so sloppy code doesn't compile.
- `add_subdirectory(tests)` / `add_subdirectory(examples)` pull in the
  `CMakeLists.txt` files in those folders.

In CMake, a **target** is a thing being built (a library or a program).
`target_link_libraries(my_program PRIVATE linegebra::linegebra)` means "my
program uses linegebra", and CMake automatically passes along the include
path and C++20 setting.

## What are tests, and what is Catch2?

A **test** is a small piece of code that checks your code gives the right
answer, for example:

```cpp
TEST_CASE("2x2 determinant") {
    Matrix<double> A{{1, 2}, {3, 4}};
    REQUIRE(det(A) == Approx(-2.0));
}
```

You run all tests after every change. If something you changed broke
something else, a test fails and tells you immediately, instead of you
discovering it weeks later.

**Catch2** is a popular C++ testing framework that provides `TEST_CASE`,
`REQUIRE`, and helpers for comparing floating-point numbers.
`tests/CMakeLists.txt` downloads it automatically the first time you
configure (`FetchContent`), so you don't have to install it.

To add tests for a new feature: create `tests/test_matrix.cpp` and add it
to the `add_executable(linegebra_tests ...)` list in `tests/CMakeLists.txt`.

## What is CI?

**Continuous Integration** means GitHub builds and tests your code
automatically on every push and pull request. `.github/workflows/ci.yml`
tells it to do so on:

- Linux with GCC,
- Linux with Clang,
- Windows with MSVC (Microsoft's compiler),
- Linux with *sanitizers*: special compiler modes that detect memory bugs
  such as reading past the end of a vector (the kind of bug in the old
  Gauss–Seidel code).

You see the result as a green tick ✓ or red cross ✗ next to your commits
and on pull requests. Code that compiles on your machine but not on another
compiler gets caught here.

## Building and testing on your own computer

You need: a C++20 compiler (GCC ≥ 10, Clang ≥ 12 or Visual Studio 2019+),
CMake ≥ 3.21 and git.

From the repository folder:

```bash
cmake -S . -B build          # 1. configure: generate build files into build/
cmake --build build          # 2. build: compile tests and examples
ctest --test-dir build       # 3. run all tests
./build/examples/hello_linegebra   # run the example program
```

(On Windows with Visual Studio, the example is at
`build\examples\Debug\hello_linegebra.exe`, and use
`ctest --test-dir build -C Debug`.)

The `build/` folder holds generated files only. It is in `.gitignore`, and
you can delete it at any time to start fresh.

After you edit code, repeat steps 2 and 3. Step 1 is only needed again if
you change a `CMakeLists.txt` (and CMake usually re-runs it for you).

Many editors (VS Code with the CMake Tools extension, CLion, Visual Studio)
open a CMake project directly and give you build/run/test buttons.

## Your everyday workflow

1. Create a branch for a feature: `git switch -c matrix-class`.
2. Add or change headers in `include/linegebra/`.
3. Write tests in `tests/` for what you added.
4. Build and run the tests locally until they pass.
5. Commit, push, and open a pull request. CI runs automatically.
6. When CI is green, merge into `main`.

## Formatting

`.clang-format` describes the code style (indentation, line length...). If
you install `clang-format` (most editors can run it on save), it formats
your code consistently so you never have to think about it.
