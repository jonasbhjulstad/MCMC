#include <SMC_Particle.hpp>
#include <SMC_Resampling.hpp>
#include <gtest/gtest.h>
#include <numeric>
#include <oneapi/dpl/random>

const size_t N_particles = 10;
using _Engine = oneapi::dpl::default_engine;

static _Engine engine;
TEST(SMC_ResamplingTest, MultinomialResample) {
  std::vector<SMC::Particle<double>> particles(N_particles);
  std::for_each(particles.begin(), particles.end(), [&](auto &particle) {
    particle.log_weight = -std::numeric_limits<double>::infinity();
  });
  particles.back().log_weight = 10;
  particles.back().state.push_back(1);

  multinomial_resample<double, _Engine>(particles, engine);
  std::for_each(particles.begin(), particles.end(),
                [&](auto &particle) { EXPECT_EQ(particle.state.size(), 1); });
}
