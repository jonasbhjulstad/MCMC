#ifndef SYCL_MCMC_RESAMPLING_HPP
#define SYCL_MCMC_RESAMPLING_HPP
#include <CL/sycl.hpp>
#include <Sycl_MCMC/random.hpp>
#include <inttypes.h>
namespace Sycl_MCMC::Math
{
    template <size_t N, typename RNG = Sycl_MCMC::random::default_rng, typename dType = float>
    static std::array<dType, N> multinomial_resample(const std::array<dType, N>& weights, RNG& rng)
    {
        Sycl_MCMC::random::multinomial_distribution<dType, N> dist(weights);
        return dist(rng);
    }

    template <size_t N, typename RNG = Sycl_MCMC::random::default_rng, typename dType = float, typename uI_t = uint32_t>
    static void multinomial_resample(sycl::queue& q, sycl::buffer<dType, 1>& particle_buf, sycl::buffer<dType, 1>& log_weight_buf, sycl::buffer<RNG, 1>& rng_buf)
    {
        sycl::buffer<dType, 1> weights(log_weight_buf.get_count());
        q.submit([&](sycl::handler& cgh) {
            auto weights_acc = weights.get_access<sycl::access::mode::write>(cgh);
            auto log_weight_acc = log_weight_buf.get_access<sycl::access::mode::read>(cgh);
            cgh.parallel_for<class log_weights_to_weights>(sycl::range<1>(log_weight_buf.get_count()), [=](sycl::id<1> idx) {
                weights_acc[idx] = sycl::exp(log_weight_acc[idx]);
            });
        });
        Sycl_MCMC::random::multinomial_distribution<dType, N> dist(q, weights, rng_buf);
        auto N_resampled = dist(q);

        //rearrange particles, weights according to N_resampled
        sycl::buffer<dType, 1> new_weights(log_weight_buf.get_count());
        sycl::buffer<dType, 1> new_particles(particle_buf.get_count());
        q.submit([&](sycl::handler& cgh) {
            auto new_weights_acc = new_weights.get_access<sycl::access::mode::write>(cgh);
            auto new_particles_acc = new_particles.get_access<sycl::access::mode::write>(cgh);
            auto weights_acc = weights.get_access<sycl::access::mode::read>(cgh);
            auto particles_acc = particle_buf.get_access<sycl::access::mode::read>(cgh);
            cgh.parallel_for<class rearrange_particles>(sycl::range<1>(log_weight_buf.get_count()), [=](sycl::id<1> idx) {
                uI_t offset = 0;
                for (size_t i = 0; i < idx[0]; i++) {
                    offset += N_resampled[i];
                }
                new_weights_acc[idx] = weights_acc[idx];
                for (size_t i = 0; i < N_resampled[idx[0]]; i++) {
                    new_particles_acc[offset + i] = particles_acc[idx];
                }
            });
        });

        particle_buf = new_particles;
        log_weight_buf = new_weights;
    }
}