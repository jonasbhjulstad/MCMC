#ifndef cpp_MCMC_SAMPLER_cpp
#define cpp_MCMC_SAMPLER_cpp
#include "SMC_Particle.hpp"
#include "SMC_Sampler.hpp"
#include <algorithm>
#include <numeric>
#include <vector>
namespace MCMC {
template <class Derived_SMC_Model, typename realtype, class _RNG_Engine>
class Sampler {
  public:
  using SMC_Model = SMC::Model<Derived_SMC_Model, realtype>;
  using SMC_Sampler = SMC::Sampler<Derived_SMC_Model, _RNG_Engine>;
  std::vector<realtype> param_current;
  std::vector<realtype> param_prop;
  std::vector<realtype> param_prev;
  std::vector<realtype> log_sum_weights;
  SMC_Model &model;
  SMC_Sampler &smc_sampler;
  _RNG_Engine &engine = smc_sampler.engine;
  size_t MCMC_iter = 0;

  Sampler(const std::vector<realtype> &param_init, const size_t N_MCMC_iters, SMC_Sampler &smc_sampler)
      : param_current(param_init), param_prev(param_init),
        param_prop(param_init), engine(smc_sampler.engine),
        smc_sampler(smc_sampler), model(smc_sampler.model) {
    log_sum_weights.resize(N_MCMC_iters);
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
      param_current.insert(param_current.begin(), param_prop.begin(),
                           param_prop.end());
    }
    // Reject the proposal
    else {
      log_sum_weights[MCMC_iter] = log_sum_weight_prev;
      param_current.insert(param_current.begin(), param_prev.begin(),
                           param_prev.end());
    }
    param_prev.insert(param_prev.begin(), param_current.begin(), param_current.end());
    // Draw new proposal sample
    model.proposal_sample(param_current, param_prop);

    // Ensure that parameters are positive by running abs(param_prop)
    std::for_each(param_prop.begin(), param_prop.end(),
                  [](realtype &p) { p = std::abs(p); });
  }

  void advance() {
    log_sum_weights[MCMC_iter] = smc_sampler.run_trajectory(param_prop);
    metropolis();
    smc_sampler.reset_particles();
    MCMC_iter++;
  }

  std::vector<realtype> run_chain()
  {
    std::for_each(log_sum_weights.begin(), log_sum_weights.end(), 
    [&](realtype& w) {w = smc_sampler.run_trajectory(param_prop);
    metropolis();
    smc_sampler.reset_particles();
    });
  }

  void reset() {
    MCMC_iter = 0;
    smc_sampler.reset_particles();
  }
};
} // namespace MCMC

#endif