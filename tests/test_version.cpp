// A first test, mostly to prove that the build, the test framework and CI
// work. Real tests arrive with the Matrix class.

#include <catch2/catch_test_macros.hpp>

#include <linegebra/linegebra.hpp>

TEST_CASE("version numbers are available", "[version]") {
    REQUIRE(linegebra::version_major == 0);
    REQUIRE(linegebra::version_minor == 0);
    REQUIRE(linegebra::version_patch >= 1);
}
