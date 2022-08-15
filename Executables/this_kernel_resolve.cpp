#include <oneapi/dpl/execution>
template <typename D>
struct Base
{
    void foo()
    {
        static_cast<D*>(this)->foo();
    }
};

struct Derived: public Base<Derived>
{
    void foo()
    {
        // ...
    }
};

int main()
{
    std::array<int, 2> arr = {1, 2};
    std::for_each(std::execution::par, arr.begin(), arr.end(), [](int i) {
        Derived a;
        a.foo();
    });
}