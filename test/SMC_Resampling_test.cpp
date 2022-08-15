#include <SMC_Particle.hpp>
#include <SMC_Resampling.hpp>
#include <gtest/gtest.h>
#include <numeric>
#include <array>
#include <oneapi/dpl/random>

const size_t N_particles = 10;
using _Engine = oneapi::dpl::default_engine;
static constexpr size_t Nx = 2;
static _Engine engine;
TEST(SMC_ResamplingTest, MultinomialResample) {
  std::array<SMC::Particle<double, Nx>, N_particles> particles;
  std::for_each(particles.begin(), particles.end(), [&](auto &particle) {
    particle.log_weight = -std::numeric_limits<double>::infinity();
  });
  particles.back().log_weight = 10;
  particles.back().state = {1, 2};

  multinomial_resample(particles, engine);
  std::for_each(particles.begin(), particles.end(),
                [&](auto &particle) { EXPECT_EQ(particle.state.size(), 2); });
}
