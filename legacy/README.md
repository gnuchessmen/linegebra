# Legacy code (2021)

The original `Matrix.h` / `Matrix.cpp` from the first attempt at linegebra.
Kept for reference only: it is not part of the build. Ideas and algorithms
from here (Gaussian elimination, Gauss–Seidel/SOR, power and inverse power
methods) will be rewritten into the new library following `ROADMAP.md`.

Known problems:

- `GaussSeidel_SOR_Solver` uses the vector `x` without ever filling it.
- `Matrix::det()` cannot handle a 1×1 matrix and uses O(n!) cofactor expansion.
- Errors are printed and an empty matrix is returned instead of throwing.
