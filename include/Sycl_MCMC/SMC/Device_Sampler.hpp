#ifndef SYCL_MCMC_SMC_SAMPLER_HPP
#define SYCL_MCMC_SMC_SAMPLER_HPP
#include <CL/sycl.hpp>
#include <Sycl_MCMC/Math/ESS.hpp>
#include <Sycl_MCMC/SMC/Model.hpp>
#include <Sycl_MCMC/random.hpp>
#include <algorithm>
#include <inttypes.h>

namespace Sycl_MCMC::SMC::Device {

template <typename Particle_State,
          typename RNG = Sycl_MCMC::random::default_rng,
          typename Model_Derived, size_t N_particles,
          typename uI_t = uint32_t, typename dType = float>
struct Sampler {
  Sampler(SMC_Model &model, sycl::queue &q, uI_t N_particles,
          dType resample_threshold = 0.5)
      : model(model)
      {
    resample_threshold = (resample_threshold < 1)
                             ? resample_threshold * N_particles
                             : resample_threshold;
  };
  ~Sampler();

  void initialize() {
    std::generate(particles.begin(), particles.end(), [&]() {
      return model.initialize();
    });
  }

  void normalize_weights() const {
    auto f_max = [](dType a, dType b) { return sycl::max(a, b); };
    auto max_weight = sycl::reduce(q, log_weight_buf, f_max);
    std::for_each(log_weights.begin(), log_weights.end(),
                  [&](dType &w) { w -= max_weight; });
  }

  dType advance() {

    std::transform(particles.begin(), particles.end(), particles.begin(), [&](Particle_State &p) {
      return model.advance(p);
    });

    std::transform(particles.begin(), particles.end(), log_weights.begin(),
                   [&](Particle_State &p) {
                     return model.log_likelihood(p);
                   });
    auto ESS = Sycl_MCMC::Math::ESS(q, log_weight_buf);
    if (ESS < resample_threshold) {
      resample(q, log_weight_buf, rng_buf);
    }

    auto ll_sum = sycl::reduce(h, log_weight_acc, sycl::plus<dType>());
    return ll_sum / N_particles;
  }

  void reset() {
    std::generate(particles.begin(), particles.end(), [&]() {
      return model.initialize();
    });
    std::transform(particles.begin(), particles.end(), log_weights.begin(),
                   [&](Particle_State &p) {
                     return model.log_likelihood(p);
                   });
  }

    dType get_resample_threshold() const { return resample_threshold; }

  private:
    dType resample_threshold;
    std::array<Particle_State, N_particles> particles;
    std::array<dType, N_particles> log_weights;
    Sycl_MCMC::SMC::Model<Particle_State, Particle_Param, Model_Derived, dType, rs_device> model;
  };
} // namespace Sycl_MCMC::SMC