#ifndef SMC_SAMPLER_hpp
#define SMC_SAMPLER_hpp
#include "MCMC_Distributions.hpp"
#include "SMC_Model.hpp"
#include "SMC_Resampling.hpp"
#include "SMC_Particle.hpp"
#include <algorithm>
#include <limits>
#include <oneapi/dpl/cmath>
#include <oneapi/dpl/random>
#include <vector>
namespace SMC {

template <class Impl_Model, class _RNG_Engine> class Sampler {
public:
  using realtype = typename Impl_Model::realtype;
  using SMC_Model = SMC::Model<Impl_Model, realtype>;

  std::vector<Particle<realtype>> particles;
  SMC_Model &model;
  realtype resample_threshold;
  size_t Nt;
  size_t t_current = 0;
  _RNG_Engine &engine;
  const size_t N_particles;

  Sampler(SMC_Model &model, const realtype threshold, const size_t N_particles, const size_t Nt,
          _RNG_Engine &engine)
      : model(model), engine(engine), N_particles(N_particles), Nt(Nt) {
    particles.resize(N_particles);
    std::for_each(particles.begin(), particles.end(),
    [&model](auto& particle){particle.state.resize(model.get_Nx());});
    reset_particles();
    resample_threshold =
        (resample_threshold < 1) ? threshold * N_particles : threshold;
  }

  void normalize_accumulate_weights() {
    realtype maxWeight = -std::numeric_limits<realtype>::infinity();
    std::for_each(
        particles.begin(), particles.end(), [&](Particle<realtype> &particle) {
          maxWeight = std::max(particle.log_weight, maxWeight);
        });

    std::for_each(
        particles.begin(), particles.end(),
        [&](Particle<realtype> &particle) { particle.log_weight -= maxWeight; });
  }

  void resample() {
    using namespace oneapi::dpl;

  }

  realtype advance(const std::vector<realtype> &param_prop) {
    normalize_accumulate_weights();
    move_particles(param_prop);

    realtype ESS = get_effective_sample_size();
    if (ESS < resample_threshold) {
      resample();
    }
    // Calculate the average logweight:
    realtype ll_sum = 0;
    std::for_each(particles.begin(), particles.end(),
                  [&](const Particle<realtype> &p) { ll_sum += p.log_weight; });
    realtype log_average_weight = ll_sum / N_particles;

    return log_average_weight;
  }

  realtype run_trajectory(const std::vector<realtype>& param_prop)
  {
    realtype log_average_weight = 0;
    for (size_t i = 0; i < Nt; ++i)
    {
      log_average_weight += advance(param_prop);
    }
    return log_average_weight;
  }

  realtype get_effective_sample_size() {
    using namespace oneapi::dpl;
    realtype sum = 0;
    realtype sumsq = 0;

    std::for_each(particles.begin(), particles.end(),
                  [&](const Particle<realtype> &p) {
                    sum += p.log_weight;
                    sumsq += p.log_weight * p.log_weight;
                  });
    return std::exp(-std::log(sumsq) + 2 * std::log(sum));
  }

  void move_particles(const std::vector<realtype> &param_prop) {

    std::for_each(particles.begin(), particles.end(), [&](Particle<realtype> &p) {
      model.advance(t_current, p.state, param_prop);
      p.log_weight += model.log_likelihood(p.state, t_current);
    });
  }

  void reset_particles() {
    std::for_each(particles.begin(), particles.end(), [&](Particle<realtype> &p) {
      model.reset(p.state);
      p.log_weight = 0;
    });
    t_current = 0;
  }

  std::vector<Particle<realtype>> get_particles() { return particles; }
};
} // namespace SMC
#endif
