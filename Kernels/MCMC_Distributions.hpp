#ifndef MCMC_DISTRIBUTIONS_HPP
#define MCMC_DISTRIBUTIONS_HPP
#include <oneapi/dpl/random>
namespace MCMC {
class multinomial_distribution {
  const size_t N;
  const size_t K;

public:
  multinomial_distribution(const size_t N, const size_t K)
      : N(N), K(K) {}

  template <typename realtype, class _Engine>
  std::vector<size_t> operator()(const std::vector<realtype>& p, _Engine& __engine) {
    std::vector<size_t> result(N);
    for (uint k = 0; k < K; k++) {
      oneapi::dpl::bernoulli_distribution bernoulli_trial(p[k]);
      result[k] = 0;
      for (uint i = 0; i < N; i++) {
        result[k] += bernoulli_trial(__engine);
      }
    }
    return result;
  }
};

} // namespace MCMC

#endif