#include <iostream>
#include "tests.hpp"
#include <vector>
#include <cstdio>
#include <string>
#include <cmath>

// Creates 2d vector of vectors by passed path
template<typename T>
std::vector<std::vector<T>> matrixFromCsv(std::string path) {

}

// If something is not working - tell it is just simplest version possible with very stupid error
int main()
{
    // Много разных тестов на разных выборках:

    // Классификация:
    xorTesting();
    lineSepTesting();

    // Регрессия:
    identicalTesting();     // y = x
    linearTesting();    // y = 2x + 1
    rootTesting();      // y = sqrt(x)
    squareTesting();    // y = sqr(x)

    return 0;
}
