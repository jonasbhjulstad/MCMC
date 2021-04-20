#include <iostream>
#include <array>
std::array <int, 5> a{1,2,3,4,5};

int main()
{
    for (int i; i < 5; i++)
    {
        std::cout << a[i];
    }
    std::cout << "hello" ;
}