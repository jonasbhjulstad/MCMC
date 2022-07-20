#ifndef PARTICLE_hpp
#define PARTICLE_hpp
#include <vector>
namespace SMC
{
template <typename realtype>
struct Particle
{
    realtype log_weight;
    std::vector<realtype> state;

    void operator=(const Particle& other)
    {
        log_weight = other.log_weight;
        state.insert(state.begin(), other.state.begin(), other.state.end());
    }
};
}
#endif