#ifndef SMC_SAMPLER_hpp
#define SMC_SAMPLER_hpp
#include "MCMC_Distributions.hpp"
#include "SMC_Model.hpp"
#include "SMC_Particle.hpp"
#include "SMC_Resampling.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/cmath>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/random>
namespace SMC {

template <class Impl_Model, size_t Nt, size_t N_particles, class _RNG_Engine>
class Sampler {
public:
  using realtype = typename Impl_Model::realtype;
  using SMC_Model =
      SMC::Model<Impl_Model, Impl_Model::Nx, Impl_Model::Np, realtype>;
  using Particle_t = Particle<realtype, SMC_Model::Nx>;
  static constexpr size_t Np = SMC_Model::Np;
  std::array<Particle_t, N_particles> particles;
  Impl_Model model;
  realtype resample_threshold;
  size_t t_current = 0;

  Sampler(Impl_Model &model, const realtype threshold)
      : model(model) {
    reset_particles();
    resample_threshold =
        (resample_threshold < 1) ? threshold * N_particles : threshold;
  }

  void normalize_accumulate_weights() {
    realtype maxWeight = -std::numeric_limits<realtype>::infinity();
    std::for_each(particles.begin(), particles.end(),
                  [&](Particle_t &particle) {
                    maxWeight = std::max(particle.log_weight, maxWeight);
                  });

    std::for_each(
        particles.begin(), particles.end(),
        [&](Particle_t &particle) { particle.log_weight -= maxWeight; });
  }

  void resample() { multinomial_resample(particles, engine); }

  realtype advance(const std::array<realtype, Np> &param_prop,
                   cl::sycl::queue &queue) {
    normalize_accumulate_weights();
    move_particles(param_prop, queue);

    realtype ESS = get_effective_sample_size();
    if (ESS < resample_threshold) {
      resample();
    }
    // Calculate the average logweight:
    realtype ll_sum = 0;

    std::for_each(particles.begin(), particles.end(),
                  [&](const Particle_t &p) { ll_sum += p.log_weight; });
    realtype log_average_weight = ll_sum / N_particles;

    return log_average_weight;
  }

  realtype run_trajectory(const std::array<realtype, Np> &param_prop,
                          cl::sycl::queue &queue) {
    realtype log_average_weight = 0;
    for (size_t i = 0; i < Nt; ++i) {
      log_average_weight += advance(param_prop, queue);
    }
    return log_average_weight;
  }

  realtype get_effective_sample_size() {
    using namespace oneapi::dpl;
    realtype sum = 0;
    realtype sumsq = 0;

    std::for_each(particles.begin(), particles.end(), [&](const Particle_t &p) {
      sum += p.log_weight;
      sumsq += p.log_weight * p.log_weight;
    });
    return std::exp(-std::log(sumsq) + 2 * std::log(sum));
  }

  void move_particles(const std::array<realtype, Np> &param_prop, sycl::queue& queue) {

    // std::for_each(particles.begin(), particles.end(), [param_prop,
    // *this](Particle_t &p) {
    //   model.advance(t_current, p.state, param_prop);
    //   p.log_weight += model.log_likelihood(p.state, t_current);
    // });
    std::uint64_t seed = 777;

    {
    sycl::buffer<Particle_t, N_particles> particle_buffer(particles.data(), sycl::range<1>(N_particles));
    queue.submit([&](sycl::handler& h)
    {
      auto particle_access = particle_buffer.get_access<sycl::access::mode::read_write>(h);
      h.parallel_for(sycl::range<1>(N_particles), [=](sycl::item<1> idx)
      {
        std::uint64_t offset = idx.get_linear_id();
        oneapi::dpl::minstd_rand engine(seed, offset);
        model.advance(t_current, particle_access[offset].state, param_prop, engine);
      });
    });
    } 
    for (auto &particle : particles) {
      model.advance(t_current, particle.state, param_prop);
      particle.log_weight += model.log_likelihood(particle.state, t_current);
    }
  }

  void reset_particles() {
    std::for_each(particles.begin(), particles.end(), [&](Particle_t &p) {
      model.reset(p.state);
      p.log_weight = 0;
    });
    t_current = 0;
  }

  std::array<Particle_t, N_particles> get_particles() { return particles; }
};
} // namespace SMC
#endif
