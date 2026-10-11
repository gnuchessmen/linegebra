# Each example is a small program showing how to use the library.
add_executable(hello_linegebra hello_linegebra.cpp)
target_link_libraries(hello_linegebra PRIVATE linegebra::linegebra linegebra_warnings)
