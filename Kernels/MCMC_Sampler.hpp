#ifndef cpp_MCMC_SAMPLER_cpp
#define cpp_MCMC_SAMPLER_cpp
#include "SMC_Particle.hpp"
#include "SMC_Sampler.hpp"
#include <algorithm>
#include <numeric>
#include <vector>
namespace MCMC {
template <class Derived_SMC_Model, size_t N_particles, size_t N_MCMC,
          typename realtype, class _RNG_Engine>
class Sampler {
public:
  static constexpr size_t Nx = Derived_SMC_Model::Nx;
  static constexpr size_t Np = Derived_SMC_Model::Np;
  static constexpr size_t Nt = Derived_SMC_Model::Nt;
  using SMC_Model = SMC::Model<Derived_SMC_Model, Nx, Np, realtype>;
  using SMC_Sampler =
      SMC::Sampler<Derived_SMC_Model, Nt, N_particles, _RNG_Engine>;
  std::array<realtype, Np> param_current;
  std::array<realtype, Np> param_prop;
  std::array<realtype, Np> param_prev;
  std::array<std::array<realtype, Np>, N_MCMC> param_list;
  std::array<realtype, N_MCMC> log_sum_weights;
  SMC_Model &model;
  SMC_Sampler &smc_sampler;
  size_t MCMC_iter = 0;
  sycl::queue& queue;

  Sampler(const std::array<realtype, Np> &param_init, SMC_Sampler &smc_sampler, sycl::queue& queue)
      : param_prop(param_init), smc_sampler(smc_sampler),
        model(smc_sampler.model), queue(queue) {
    param_list[0] = param_init;
    log_sum_weights[0] = std::numeric_limits<realtype>::infinity();
    log_sum_weights[1] = std::numeric_limits<realtype>::infinity();
  }

  void metropolis() {
    // Did the proposal likelihood improve fit?
    realtype log_sum_weight_prev =
        (MCMC_iter == 0) ? log_sum_weights[0] : log_sum_weights[MCMC_iter - 1];

    realtype alpha_prop = exp(log_sum_weights[MCMC_iter] - log_sum_weight_prev);
    realtype alpha_Metropolis = (1 < alpha_prop) ? 1 : alpha_prop;
    oneapi::dpl::uniform_real_distribution<realtype> uniform_dist;
    // Accept the proposal
    if (uniform_dist(engine) < alpha_Metropolis) {
      param_list[MCMC_iter] = param_prop;
    }
    // Reject the proposal
    else {
      log_sum_weights[MCMC_iter] = log_sum_weight_prev;
      param_list[MCMC_iter] = param_list[MCMC_iter - 1];
    }
    // Draw new proposal sample
    model.proposal_sample(param_list[MCMC_iter], param_prop);

    // Ensure that parameters are positive by running abs(param_prop)
    std::for_each(param_prop.begin(), param_prop.end(),
                  [](realtype &p) { p = std::abs(p); });
  }

  void advance() {
    log_sum_weights[MCMC_iter] = smc_sampler.run_trajectory(param_prop, queue);
    metropolis();
    smc_sampler.reset_particles();
    MCMC_iter++;
  }

  std::array<std::array<realtype, Np>, N_MCMC> run_chain(bool verbose = false) {
    size_t iter = 0;
    for (auto &weight : log_sum_weights) {
      weight = smc_sampler.run_trajectory(param_prop, queue);
      if (verbose && iter % (log_sum_weights.size() / 10) == 0) {
        std::cout << "Iter: " << iter << " log_sum_weights: " << weight << std::endl;
      }
    iter++;
    metropolis();
    MCMC_iter++;
    smc_sampler.reset_particles();
    }
    // size_t iter = 0;
    // std::for_each(log_sum_weights.begin(), log_sum_weights.end(),
    // [&](realtype& w) {
    //   w = smc_sampler.run_trajectory(param_prop);
    //   if (verbose && iter % (log_sum_weights.size()/10) == 0) {
    //     std::cout << "Iter: " << iter << " log_sum_weights: " << w <<
    //     std::endl;
    //   }

  return param_list;
}

  void reset() {
  MCMC_iter = 0;
  smc_sampler.reset_particles();
}
};
} // namespace MCMC

#endif