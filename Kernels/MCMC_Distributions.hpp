#ifndef MCMC_DISTRIBUTIONS_HPP
#define MCMC_DISTRIBUTIONS_HPP
#include <array>
#include <oneapi/dpl/random>
#include <oneapi/mkl.hpp>
namespace MCMC {

template <typename realtype, size_t N>
constexpr std::array<realtype, N> zeroArray{[]()
{
  std::array<realtype, N> res{};
  for (int i = 0; i < N; i++)
  {
    res[i] = 0;
  }
  return res;}()
};

template <typename realtype, class _Engine, size_t N, size_t K>
std::array<size_t, N> multinomial_sample(const std::array<realtype, K> &p,
                                         _Engine &__engine) {
  std::array<size_t, N> result;
  for (uint k = 0; k < K; k++) {
    oneapi::dpl::bernoulli_distribution bernoulli_trial(p[k]);
    result[k] = 0;
    for (uint i = 0; i < N; i++) {
      result[k] += bernoulli_trial(__engine);
    }
  }
  return result;
}

class binomial_distribution {
  size_t N;
  oneapi::dpl::bernoulli_distribution<bool> dist;

public:
  template <typename realtype>
  binomial_distribution(const size_t N, const realtype p) : N(N), dist(p) {}
  template <typename _Engine> size_t operator()(_Engine &engine) {
    size_t res = 0;
    for (int i = 0; i < N; i++) {
      res += dist(engine);
    }
    return res;
  }
};

// Multivariate normal distribution using the LLT-decomposition
template <typename realtype, size_t N> class multivariate_normal_distribution {
  std::array<realtype, N> mu;
  std::array<realtype, N*N> Sigma;

public:
  multivariate_normal_distribution(const std::array<realtype, N*N> &_Sigma, sycl::queue& queue) {
    for (size_t i = 0; i < N * N; i++) {
      Sigma[i] = _Sigma[i];
    }
    int temp_size = oneapi::mkl::lapack::potrf_scratchpad_size<realtype>(
        queue, oneapi::mkl::uplo::upper, N, 0);
    realtype *temp = new realtype[temp_size];

    // Get cholesky decomposition of sigma using oneapi::mkl::lapack::potrf
    oneapi::mkl::lapack::potrf(queue, oneapi::mkl::uplo::upper, N,
                               Sigma.data(), 0, temp, temp_size);
  }

  template <typename _Engine>
  std::array<realtype, N> operator()(_Engine &engine, const std::array<realtype, N> &mu) {
    oneapi::dpl::normal_distribution<realtype> normal_dist;
    std::array<realtype, N> z;
    for (auto &zk : z) {
      zk = normal_dist(engine);
    }
    using namespace oneapi::mkl::lapack;
    // get the Cholesky decomposition of Sigma

    std::array<realtype, N> res;
    for (int i = 0; i < N; i++) {
      res[i] = mu[i];
    }
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        res[i] += z[j] * Sigma[i * N + j];
      }
    }
    return res;
  }
};

template <typename realtype, size_t N, class _Engine>
std::array<realtype, N> discrete_sample(const std::array<realtype, N> &p,
                                        _Engine &engine) {
  std::array<realtype, N> result;
  std::fill(result.begin(), result.end(), 0);
  std::array<realtype, p> cumsum;
  cumsum[0] = p[0];
  for (size_t i = 1; i < N; i++) {
    cumsum[i] = cumsum[i - 1] + p[i];
  }
  // sample from uniform distribution
  oneapi::dpl::uniform_real_distribution<realtype> uniform_dist(0, 1);
  for (size_t i = 0; i < N; i++) {
    realtype u = uniform_dist(engine);
    for (size_t j = 0; j < N; j++) {
      if (u < cumsum[j]) {
        result[j]++;
        break;
      }
    }
  }

  return result;
}
template <typename T>
T normal_distribution_pdf(T x, T mu, T sigma)
{
    static const T inv_sqrt_2pi = 0.3989422804014327;
    T a = (x - mu) / sigma;

    return inv_sqrt_2pi / sigma * std::exp(-T(0.5) * a * a);
}


} // namespace MCMC

#endif