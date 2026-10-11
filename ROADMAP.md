# linegebra roadmap

A plan for turning linegebra into a complete, well-tested dense linear algebra
library in modern C++, built in milestones that each produce something usable.

## 1. Goals and non-goals

**Goals**

- Dense matrices and vectors over `float`, `double` and `std::complex<T>`.
- A clean, hard-to-misuse API: `C = A * B`, `x = solve(A, b)`, `auto [Q, R] = qr(A)`.
- Correctness first: every algorithm is tested against known answers and
  mathematical identities (e.g. `Q * R ≈ A`, `Q^T Q ≈ I`).
- Reasonable performance for small to medium matrices (up to ~1000×1000),
  without heroics.
- A project that teaches modern C++ along the way.

**Non-goals (explicitly out of scope for 1.0)**

- Sparse matrices, GPU/SIMD intrinsics, multithreading.
- Expression templates / lazy evaluation.
- Competing with BLAS/LAPACK/Eigen on speed.
- Tensors (rank > 2) and arbitrary precision arithmetic.

Writing these down is what keeps the project finishable. Revisit them only
after 1.0.

## 2. Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Language standard | C++20 | Concepts give readable template errors; `std::span`, `<=>`, designated initializers. |
| Build system | CMake (≥ 3.20) | The de facto standard; easy for others to consume via `FetchContent`/`find_package`. |
| Library form | Header-only, templated on scalar type | Templates must live in headers anyway; zero-friction to use. |
| Storage | One contiguous `std::vector<T>`, row-major, plus `rows`, `cols` | Cache friendly, cheap to copy/move, no ragged rows (the old `vector<vector<double>>` could be ragged). |
| Vectors | `Vector<T>` as its own type (column vector) | Clearer API and errors than a "matrix with one column". |
| Error handling | Exceptions for invalid arguments (`linegebra::dimension_error`, `linegebra::singular_matrix_error`), `assert`-style bounds checks in debug builds | Library code must never `std::cout` an error and return garbage. |
| Iterative algorithms | Return a result struct (`solution`, `iterations`, `residual`, `converged`) | Callers decide what to print; no hidden I/O. |
| Namespace | `linegebra` | Matches the project name. |
| Tests | Catch2 v3 via `FetchContent` | Simple syntax, built-in floating-point matchers. |
| CI | GitHub Actions: GCC + Clang on Linux, MSVC on Windows, ASan/UBSan job | Catch portability and memory bugs early. |

## 3. Repository layout

```
linegebra/
├── CMakeLists.txt
├── include/linegebra/
│   ├── linegebra.hpp        # umbrella header
│   ├── matrix.hpp           # Matrix<T>
│   ├── vector.hpp           # Vector<T>
│   ├── ops.hpp              # arithmetic, norms, products
│   ├── decompositions/      # lu.hpp, cholesky.hpp, qr.hpp, eigen.hpp, svd.hpp
│   ├── solvers/             # direct.hpp, iterative.hpp
│   └── errors.hpp
├── tests/                   # one test file per header
├── examples/                # small runnable programs
├── benchmarks/              # optional, Milestone 7
└── docs/
```

## 4. Milestones

Each milestone ends with: all tests green in CI, examples updated, a git tag
(`v0.1.0`, `v0.2.0`, ...), and a short CHANGELOG entry. Work in small
branches, one feature per pull request.

### Milestone 0: Project foundation

- CMake project with an `INTERFACE` library target `linegebra::linegebra`.
- Catch2 test target, `ctest` integration.
- `.clang-format`, compiler warnings as errors (`-Wall -Wextra -Wpedantic`).
- GitHub Actions CI.
- Move the 2021 `Matrix.h`/`Matrix.cpp` into `legacy/` for reference.

*Done when:* an empty test passes in CI on all three compilers.

### Milestone 1: Core containers (`v0.1`)

- `Matrix<T>`: construct from `(rows, cols)`, `(rows, cols, fill)`, nested
  initializer list `{{1, 2}, {3, 4}}` (rejecting ragged input).
- `Vector<T>`: construct from size or initializer list.
- Element access: `A(i, j)`, `v[i]`, checked `at()`.
- Shape queries: `rows()`, `cols()`, `size()`, `is_square()`, `empty()`.
- Factories: `zeros`, `ones`, `identity`, `diag(vector)`, `random(rows, cols, seed)`.
- Row/column/block extraction and assignment (copying, not views, for now).
- `operator<<` for printing; approximate equality `approx_equal(A, B, tol)`.
- Iterators (`begin`/`end`) so standard algorithms work.

*C++ to learn:* classes, RAII and the "rule of zero", `const` correctness,
references, `std::initializer_list`, exceptions.

### Milestone 2: Arithmetic and basic operations (`v0.2`)

- `+`, `-`, unary `-`, scalar `*` and `/`, compound `+=`, `-=`, `*=`, `/=`.
- Matrix-matrix, matrix-vector products.
- `transpose`, `conj_transpose` (for complex), `trace`.
- Element-wise (Hadamard) product and division.
- `dot`, `cross` (3-vectors), `outer`.
- Norms: vector 1, 2, ∞; matrix 1, ∞, Frobenius.
- Integer `pow(A, n)` via repeated squaring (replaces the old `operator^`, which
  has the wrong precedence in C++).

*C++ to learn:* operator overloading (member vs. free), templates, concepts
(`template <Scalar T>`), move semantics.

### Milestone 3: Direct solvers and decompositions (`v0.3`)

- Triangular solves (forward/back substitution).
- **LU with partial pivoting** → `solve(A, b)`, `det(A)`, `inverse(A)`.
  Replaces the old recursive cofactor `det()`, which is O(n!).
- **Cholesky** for symmetric positive-definite matrices.
- **QR via Householder reflections** → least-squares `lstsq(A, b)`.
- `rank` (via QR with column pivoting, or postponed to SVD).
- Structured returns: `auto [L, U, P] = lu(A);`.

*Test with:* reconstruction identities, Hilbert matrices (ill-conditioned),
singular matrices must throw.

*C++ to learn:* structured bindings, returning structs, numerical tolerances.

### Milestone 4: Iterative solvers (`v0.4`)

- Jacobi, Gauss–Seidel, SOR (port from the 2021 code, fixing the
  uninitialized `x` bug).
- Conjugate Gradient for symmetric positive-definite systems.
- Common options struct: `{.tol = 1e-10, .max_iter = 1000}`, and result struct
  reporting convergence instead of printing it.

### Milestone 5: Eigenvalues and SVD (`v0.5`)

- Power and inverse power iteration (port), plus Rayleigh quotient.
- **Symmetric eigensolver:** Jacobi eigenvalue algorithm (simple and robust).
- **SVD:** one-sided Jacobi. Then `pinv`, `cond`, `rank`, matrix 2-norm.
- Stretch: general (non-symmetric) eigenvalues via Hessenberg reduction and
  shifted QR iteration, returning complex eigenvalues.

This is the hardest milestone. Use a textbook (Golub & Van Loan, *Matrix
Computations*; or Trefethen & Bau, *Numerical Linear Algebra*) as the
reference for each algorithm.

### Milestone 6: Complex numbers and generality (`v0.6`)

- Verify every algorithm works with `std::complex<float/double>`
  (conjugate transposes in QR, Hermitian Cholesky, etc.).
- Mixed scalar operations where sensible (`double * Matrix<complex<double>>`).
- Optional: non-owning `MatrixView<T>` for blocks without copying.

### Milestone 7: Performance and release (`v1.0`)

- Benchmarks (Google Benchmark) for matmul, LU, QR at several sizes.
- Cache-friendly matmul (i-k-j loop order, then blocking).
- Profile and remove unnecessary copies.
- Documentation: README with quick start, API reference (Doxygen), examples
  for each feature.
- Installable via CMake `install()` + `find_package(linegebra)`.

## 5. Feature checklist for 1.0

- [ ] Matrix/Vector containers, factories, element and block access
- [ ] Arithmetic, products, transpose, trace, norms
- [ ] LU, Cholesky, QR
- [ ] `solve`, `det`, `inverse`, `lstsq`
- [ ] Jacobi, Gauss–Seidel, SOR, Conjugate Gradient
- [ ] Power/inverse iteration, symmetric eigensolver
- [ ] SVD, `pinv`, `rank`, `cond`
- [ ] `float`, `double`, `complex` support
- [ ] Tests for everything, CI on 3 compilers, docs, examples

## 6. Target API sketch

```cpp
#include <linegebra/linegebra.hpp>
namespace la = linegebra;

la::Matrix<double> A{{4, 1}, {1, 3}};
la::Vector<double> b{1, 2};

auto x          = la::solve(A, b);          // LU under the hood
auto [L, U, P]  = la::lu(A);
auto [Q, R]     = la::qr(A);
auto eig        = la::eigh(A);              // eig.values, eig.vectors
auto [U2, S, V] = la::svd(A);

auto r = la::conjugate_gradient(A, b, {.tol = 1e-12});
if (r.converged) std::cout << r.solution << '\n';
```

## 7. Lessons from the 2021 attempt

- Plan scope first; the feature list above is the contract.
- Library code never prints. Return values, throw on errors.
- Contiguous storage, not `vector<vector<double>>`.
- Pass large objects by `const&`; rely on move semantics for returns.
- Write the test before (or alongside) the algorithm.
- Choose algorithms with good complexity (LU determinant, not cofactor
  expansion).
