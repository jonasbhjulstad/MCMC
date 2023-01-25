#ifndef SYCL_MCMC_MCMC_MODEL_HPP
#define SYCL_MCMC_MCMC_MODEL_HPP

#include <CL/sycl.hpp>
#include <Sycl_MCMC/SMC/Model.hpp>

namespace Sycl_MCMC::MCMC
{
template <typename Param, typename Derived, typename dType = float>
struct Model
{
    Particle_Param proposal_sample(const Particle_Param &param) const
    {
        return static_cast<const Derived *>(this)->proposal_sample(param);
    }

    dType log_likelihood(const Param &param) const
    {
        return static_cast<const Derived *>(this)->log_likelihood(param);
    }
};




} // namespace Sycl_MCMC::MCMC

#endif