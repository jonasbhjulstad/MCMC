#ifndef SMC_MODEL_HPP
#define SMC_MODEL_HPP
#include <stddef.h>
#include <array>
namespace SMC{

template <typename Derived, size_t _Nx, size_t _Np, typename dtype>
struct Model
{
    static constexpr size_t Nx = _Nx;
    static constexpr size_t Np = _Np;
    using realtype = dtype;
    //Should integrate the state & update particle weight
    inline void advance(const size_t &t, std::array<realtype, Nx>& particle_state, const std::array<realtype, Np>& param)
    {
        static_cast<Derived*>(this)->advance(t, particle_state, param);
    }

    //Update the parameter proposal:
    void proposal_sample(const std::array<realtype, Np>& param_prop, std::array<realtype, Np>& res)
    {
        static_cast<Derived*>(this)->proposal_sample(param_prop, res);
    }
    //Likelihood required for particle weight updates
    realtype log_likelihood(const std::array<realtype, Nx>& state, const size_t &t)
    {
        return static_cast<Derived*>(this)->log_likelihood(state, t);
    }
    void reset(std::array<realtype, Nx>& state)
    {
        static_cast<Derived*>(this)->reset(state);
    }

    constexpr size_t get_Nx()
    {
        return Nx;
    }
};
}


#endif