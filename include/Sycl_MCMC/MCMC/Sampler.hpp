#ifndef SYCL_MCMC_MCMC_SAMPLER_HPP
#define SYCL_MCMC_MCMC_SAMPLER_HPP
#include <CL/sycl.hpp>
#include <Sycl_MCMC/SMC/Model.hpp>
#include <Sycl_MCMC/MCMC/Model.hpp>
#include <Sycl_MCMC/random.hpp>
#include <inttypes.h>
#include <vector>
namespace Sycl_MCMC::MCMC
{
template <typename Theta, typename Derived,
typename RNG = Sycl_MCMC::random::default_rng, typename uI_t = uint32_t, typename dType = float>
struct Sampler
{
    using Model_t = Model<Particle_Param, MCMC_D, dType>;
    // using SMC_Model_t = SMC::Model<Particle_State, Particle_Param, SMC_D, dType>;
    // using SMC_Sampler_t = Sycl_MCMC::SMC::Sampler<Particle_State, RNG, SMC_Model_t, N_particles, uI_t, dType>;
    
    Sampler(Model_t& model, SMC_Sampler_t& smc_sampler): model(model), smc_sampler(smc_sampler)
    {

    }

    Sampler(Model_t& model, SMC_Model_t& smc_model, sycl::queue& q, uI_t N_particles, dType resample_threshold = 0.5): model(model), smc_sampler(smc_model, q, N_particles, resample_threshold)
    {
    }

    Theta advance(const Theta theta)
    {
    }


    std::vector<Theta> run(uI_t N_iterations, Theta theta_0 = Theta{})
    {
        std::vector<Theta> thetas(N_iterations+1);
        thetas[0] = theta_0;
    }


    Model_t model;
    SMC_Sampler_t smc_sampler;
};
}
#endif