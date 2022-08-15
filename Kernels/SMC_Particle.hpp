#ifndef PARTICLE_hpp
#define PARTICLE_hpp
#include <stddef.h>
#include <array>
namespace SMC
{
template <typename realtype, size_t Nx>
struct Particle
{
    realtype log_weight;
    std::array<realtype, Nx> state;
};
}
#endif