#ifndef SYCL_MCMC_MODEL_HPP
#define SYCL_MCMC_MODEL_HPP
#include <CL/sycl.hpp>
#include <Sycl_MCMC/Math/Resampling.hpp>

namespace Sycl_MCMC::SMC
{
  template <typename Particle_State, typename Particle_Param>
  using Fn_Advance = Particle_State (*)(const Particle_State &state,
                                        const Particle_Param &param);

  template <typename Particle_State, typename Particle_Param>
  using Fn_Initialize = Particle_State (*)(const Particle_Param &param);

  enum class Resampling_Strategy
  {
    RESAMPLING_STRATEGY_MULTINOMIAL
  };

  template <typename Particle_State, typename Particle_Param, typename Derived, typename dType = float>
  struct Model
  {
    Particle_State advance(const Particle_State &state,
                           const Particle_Param &param) const
    {
      return static_cast<const Derived *>(this)->advance(state, param);
    }

    Particle_State initialize(const Particle_Param &param) const
    {
      return static_cast<const Derived *>(this)->initialize(param);
    }

    dType resample(sycl::queue &q, sycl::buffer<dType, 1> &log_weight_buf)
    {
      return static_cast<const Derived *>(this)->resample();
    }

    dType log_likelihood(const Particle_Param &param) const
    {
      return static_cast<const Derived *>(this)->log_likelihood(param);
    }
  };
}