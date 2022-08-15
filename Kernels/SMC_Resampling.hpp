#ifndef SMC_RESAMPLING_HPP
#define SMC_RESAMPLING_HPP
#include "SMC_Particle.hpp"
#include <algorithm>
#include <numeric>
#include <oneapi/dpl/random>
#include <oneapi/dpl/execution>
#include <array>
namespace SMC {
template <typename realtype, size_t Nx, size_t N_particles, typename _RNG_Engine>
void multinomial_resample(std::array<Particle<realtype, Nx>, N_particles> &particles,
                          _RNG_Engine &engine) {
  std::array<realtype,N_particles> resampling_weights;
  std::array<size_t,N_particles> resampling_counts;
  std::array<realtype,N_particles> weight_cumsum;
  std::array<realtype,N_particles> urandom_vec;
  std::array<Particle<realtype, Nx>, N_particles> new_particles;

  // Compute cumulative weights
  for (int i = 0; i < N_particles; ++i)
    resampling_weights[i] = std::exp(particles[i].log_weight);
  std::partial_sum(resampling_weights.begin(), resampling_weights.end(),
                   weight_cumsum.begin());
  // Generate uniform random numbers
  oneapi::dpl::uniform_real_distribution<realtype> uniform_dist(
      0, weight_cumsum.back());
  std::for_each(urandom_vec.begin(), urandom_vec.end(),
                [&](auto &val) { val = uniform_dist(engine); });

  // Get sampling counts based on the cumulative sum of weights
  std::for_each(
      urandom_vec.begin(), urandom_vec.end(), [&](const auto& rnd_val) {
        auto upper_bound = std::upper_bound(weight_cumsum.begin(),
                                             weight_cumsum.end(), rnd_val);
        size_t resample_idx = std::distance(weight_cumsum.begin(), upper_bound);
        resampling_counts[resample_idx]++;
      });

  size_t count_idx = 0;
  // Assign the new particles
  std::for_each(new_particles.begin(), new_particles.end(),
                [&](auto &new_particle) {
                  while (resampling_counts[count_idx] == 0)
                    count_idx++;
                  new_particle = particles[count_idx];
                  resampling_counts[count_idx]--;
                });

  for (int i = 0; i < N_particles; i++)
  {
    particles[i] = new_particles[i];
  }

}

} // namespace SMC

#endif // SMC_RESAMPLING_HPP