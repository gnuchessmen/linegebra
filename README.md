# linegebra

> **Archived: no longer maintained.**
>
> This project was a learning exercise in building a dense linear algebra
> library in C++. It has been retired in favour of a new project that
> implements machine learning methods from scratch in Python (linear and
> logistic regression, GLMs, SVMs, decision trees, random forests and
> boosting), using NumPy as the linear algebra layer.
>
> What remains here:
>
> - [`legacy/`](legacy/): the original 2021 C++ `Matrix` code (Gaussian
>   elimination, Gauss–Seidel/SOR, power and inverse power methods).
> - [ROADMAP.md](ROADMAP.md): the plan that was drawn up for the library.
> - A working CMake/Catch2/CI project skeleton, explained in
>   [docs/PROJECT_SETUP.md](docs/PROJECT_SETUP.md).
>
> The code is MIT licensed and free to reuse.

A header-only dense linear algebra library in modern C++ (C++20).

## Building and running the tests

Requirements: a C++20 compiler, CMake ≥ 3.21, git.

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

## Using linegebra in your own CMake project

```cmake
include(FetchContent)
FetchContent_Declare(linegebra
    GIT_REPOSITORY https://github.com/gnuchessmen/linegebra.git
    GIT_TAG        main)
FetchContent_MakeAvailable(linegebra)

target_link_libraries(your_program PRIVATE linegebra::linegebra)
```

```cpp
#include <linegebra/linegebra.hpp>
```

## License

MIT, see [LICENSE](LICENSE).
