#ifndef SYCL_MCMC_METROPOLIS_HPP
#define SYCL_MCMC_METROPOLIS_HPP
#include <CL/sycl.hpp>
#include <Sycl_MCMC/SMC/Model.hpp>
#include <Sycl_MCMC/random.hpp>
#include <vector>
#include <inttypes.h>

namespace Sycl_MCMC::MCMC {
template <typename Theta, typename RNG = Sycl_MCMC::random::default_rng,
          typename uI_t = uint32_t, typename dType = float>
Theta metropolis_step(const Theta &theta, const Sycl_MCMC::SMC::Model &model,
                      RNG &rng) {
  Theta theta_prop = model.proposal_sample(theta);
  dType log_ratio =
      model.log_likelihood(theta_prop) - model.log_likelihood(theta);
  if (log_ratio > 0) {
    return theta_prop;
  } else {
    auto dist = Sycl_MCMC::random::uniform_real_distribution<dType>(0, 1);

    return (dist(rng) < sycl::exp(log_ratio)) ? theta_prop : theta;
  }
}
} // namespace Sycl_MCMC::MCMC

#endif