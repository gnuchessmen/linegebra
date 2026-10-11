# linegebra

A header-only dense linear algebra library in modern C++ (C++20).

The library is in an early stage. See [ROADMAP.md](ROADMAP.md) for the plan
and [docs/PROJECT_SETUP.md](docs/PROJECT_SETUP.md) for how the project is
organized.

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
