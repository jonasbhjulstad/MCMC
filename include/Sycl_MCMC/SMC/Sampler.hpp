#ifndef SYCL_MCMC_SMC_SAMPLER_HPP
#define SYCL_MCMC_SMC_SAMPLER_HPP
#include <CL/sycl.hpp>
#include <Sycl_MCMC/Math/ESS.hpp>
#include <Sycl_MCMC/SMC/Model.hpp>
#include <Sycl_MCMC/random.hpp>
#include <inttypes.h>

namespace Sycl_MCMC::SMC {

template <typename Particle_State,
          typename Particle_Param,
          typename RNG = Sycl_MCMC::random::default_rng,
          typename Model_Derived,
          typename uI_t = uint32_t, typename dType = float>
struct Sampler {

  using Model_t = Sycl_MCMC::SMC::Model<Particle_State, Particle_Param, Model_Derived, dType>;
  Sampler(SMC_Model &model, sycl::queue &q, uI_t N_particles,
          dType resample_threshold = 0.5)
      : model(model), q(q), particle_buf(sycl::range<1>(N_particles)),
        log_weight_buf(sycl::range<1>(N_particles)) {
    resample_threshold = (resample_threshold < 1)
                             ? resample_threshold * N_particles
                             : resample_threshold;
  };
  ~Sampler();

  void initialize() {
    q.submit([&]() {
      auto particle_acc = particle_buf.get_access<sycl::access::mode::write>(h);
      h.parallel_for(
          sycl::range<1>(particle_buf.get_count()),
          [=](sycl::id<1> idx) { particle_acc[idx] = model.initialize(); });
    });
  }

  void normalize_weights() {
    auto f_max = [](dType a, dType b) { return sycl::max(a, b); };
    auto max_weight = sycl::reduce(q, log_weight_buf, f_max);
    q.submit([&](sycl::handler &h) {
      auto log_weight_acc =
          log_weight_buf.get_access<sycl::access::mode::read_write>(h);
      h.parallel_for(
          sycl::range<1>(log_weight_buf.get_count()),
          [=](sycl::id<1> idx) { log_weight_acc[idx] -= max_weight; });
    });
  }

  dType advance() {
    q.submit([&](sycl::handler &h) {
      auto particle_acc =
          particle_buf.get_access<sycl::access::mode::read_write>(h);
      auto log_weight_acc =
          log_weight_buf.get_access<sycl::access::mode::read_write>(h);
      h.parallel_for(
          sycl::range<1>(particle_buf.get_count()), [=](sycl::id<1> idx) {
            particle_acc[idx] = model.advance(particle_acc[idx]);
            log_weight_acc[idx] += model.log_likelihood(particle_acc[idx]);
            //global fence here
          });
    });

    auto ESS = Sycl_MCMC::Math::ESS(q, log_weight_buf);
    if (ESS < resample_threshold) {
      resample(q, log_weight_buf, rng_buf);
    }

    auto ll_sum = sycl::reduce(h, log_weight_acc, sycl::plus<dType>());
    return ll_sum / N_particles;
  }

  dType run(uI_t N_iterations)
  {
    dType ll_sum = 0;
    for (uI_t i = 0; i < N_iterations; i++)
    {
      ll_sum += advance();
    }
    return ll_sum / N_iterations;
  }

  void reset() {
    q.submit([&](sycl::handler &h) {
      auto particle_acc =
          particle_buf.get_access<sycl::access::mode::read_write>(h);
      auto log_weight_acc =
          log_weight_buf.get_access<sycl::access::mode::read_write>(h);
      h.parallel_for(
          sycl::range<1>(particle_buf.get_count()), [=](sycl::id<1> idx) {
            particle_acc[idx] = model.initialize();
            log_weight_acc[idx] = model.log_likelihood(particle_acc[idx]);
          });
    });

    dType get_resample_threshold() const { return resample_threshold; }

  private:
    dType resample_threshold;
    sycl::queue &q;
    sycl::buffer<Particle_State> particle_buf;
    sycl::buffer<dType> log_weight_buf;
    sycl::buffer<RNG, 1> rng_buf;
    Model_t model;
  };
} // namespace Sycl_MCMC::SMC