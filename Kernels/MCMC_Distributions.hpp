#ifndef MCMC_DISTRIBUTIONS_HPP
#define MCMC_DISTRIBUTIONS_HPP
#include <oneapi/dpl/random>
#include <Eigen/Dense>
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

class binomial_distribution
{
  size_t N;
  oneapi::dpl::bernoulli_distribution<bool> dist;
  public:
  template <typename realtype>
  binomial_distribution(const size_t N, const realtype p): N(N), dist(p){}
  template <typename _Engine>
  size_t operator()(_Engine& engine)
  {
    size_t res = 0;
    for (int i = 0; i < N; i++)
    {
      res += dist(engine);
    }
    return res;
  }
};


//Multivariate normal distribution using the LLT-decomposition
template <typename realtype>
class multivariate_normal_distribution
{
  using Mat = Eigen::Matrix<realtype, -1, -1>;
  using Vec = Eigen::Vector<realtype, -1>;
  Eigen::LLT<Mat> llt_decomposition;
  Vec mu;

  multivariate_normal_distribution(const Mat& Sigma)
  {
    llt_decomposition.compute(Sigma);
  }

  template <typename _Engine>
  Vec operator()(_Engine& engine, const Vec& mu)
  {
    oneapi::dpl::normal_distribution<realtype> normal_dist;
    Vec z(mu.size());
    for (auto& zk: z)
    {
      zk = normal_dist(engine);
    }
    return mu + llt_decomposition.matrixLLT().dot(z);
  }

};

} // namespace MCMC

#endif