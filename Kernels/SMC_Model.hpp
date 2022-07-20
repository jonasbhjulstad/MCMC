#ifndef SMC_MODEL_HPP
#define SMC_MODEL_HPP
#include <stddef.h>
#include <vector>
namespace SMC{

template <typename Derived, typename dtype>
struct Model
{
    using realtype = dtype;
    //Should integrate the state & update particle weight
    void advance(const size_t &t, std::vector<realtype>& particle_state, const std::vector<realtype>& param)
    {
        static_cast<Derived*>(this)->advance(t, particle_state, param);
    }

    //Update the parameter proposal:
    void proposal_sample(const std::vector<realtype>& param_prop, std::vector<realtype>& res)
    {
        static_cast<Derived*>(this)->proposal_sample(param_prop, res);
    }
    //Likelihood required for particle weight updates
    realtype log_likelihood(const std::vector<realtype> &state, const size_t &t)
    {
        return static_cast<Derived*>(this)->log_likelihood(state, t);
    }
    void reset(std::vector<realtype>& state)
    {
        static_cast<Derived*>(this)->reset(state);
    }

    size_t get_Nx()
    {
        return static_cast<Derived*>(this)->get_Nx();
    }
};
}


#endif