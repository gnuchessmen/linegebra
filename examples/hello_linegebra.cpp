// The smallest possible program that uses linegebra.

#include <iostream>

#include <linegebra/linegebra.hpp>

int main() {
    std::cout << "linegebra version "
              << linegebra::version_major << '.'
              << linegebra::version_minor << '.'
              << linegebra::version_patch << '\n';
}
