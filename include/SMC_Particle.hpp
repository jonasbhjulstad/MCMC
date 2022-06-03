#ifndef PARTICLE_H
#define PARTICLE_H
#include <SMC_Utils.clhpp>
namespace SMC
{
template <size_t Nx>
struct SMC_Particle
{
    float log_weight;
    float state[Nx];

    void operator=(const SMC_Particle& other)
    {
        log_weight = other.log_weight;
        assign_vec(state, other.state, Nx);
    }
};
}
#endif