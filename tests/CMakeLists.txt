# Download Catch2 (the test framework) at configure time.
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(Catch2)

# One executable holds all tests. Add new test files to this list.
add_executable(linegebra_tests
    test_version.cpp)

target_link_libraries(linegebra_tests PRIVATE
    linegebra::linegebra
    linegebra_warnings
    Catch2::Catch2WithMain)

# Register every TEST_CASE with CTest so `ctest` can run them individually.
include(Catch)
catch_discover_tests(linegebra_tests)
