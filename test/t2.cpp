#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/random>
#include <iostream>

int main()
{
    std::vector<int> data(1000);
    std::fill(oneapi::dpl::execution::par_unseq, data.begin(), data.end(), 42);
    std::cout << "Hello" <<std::endl;
}