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
        state.resize(other.state.size());
        for (int i = 0; i < other.state.size(); i++)
        {
            state[i] = other.state[i];
        }
        
    }
};
}
#endif