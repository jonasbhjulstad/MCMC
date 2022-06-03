#ifndef DISTRIBUTIONS_HH
#define DISTRIBUTIONS_HH
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

class Rng
{
private:
gsl_vector* vec_mu;
gsl_vector* vec_result;
gsl_matrix* mat_sigma;
gsl_rng* pRng;
long N;
public:
Rng();
Rng(const long &);
~Rng();

gsl_rng* Raw();

double GaussianLogPDF(const double &x, const double &mu, const double &std);

void GaussianMultivariate(const double* mu, const double* std, double * res);

double BetaBinomial(const double & mu, const double & nu, const double & n);

void Multinomial(long n, long k, const double* w, unsigned * X);

double Uniform(const double &a, const double &b);

};

inline Rng::Rng(){};
inline Rng::Rng(const long &Nx)
{
  N = Nx;
  vec_mu = gsl_vector_alloc(N);
  vec_result = gsl_vector_alloc(N);
  mat_sigma = gsl_matrix_alloc(N, N);
  gsl_matrix_set_all(mat_sigma, 0);
  pRng = gsl_rng_alloc(gsl_rng_taus2);
}

inline Rng::~Rng()
{
  gsl_vector_free(vec_mu);
  gsl_vector_free(vec_result);
  gsl_matrix_free(mat_sigma);
  gsl_rng_free(pRng);
}

inline gsl_rng* Rng::Raw()
{
    return pRng;
}

inline double Rng::GaussianLogPDF(const double &x, const double &mu, const double &std)
{
  return gsl_ran_gaussian_pdf(x-mu, std);
}

inline void Rng::GaussianMultivariate(const double* mu, const double* std, double * res)
{
  for (size_t i = 0; i < N; i++)
  {
    gsl_vector_set(vec_mu, i, mu[i]);
    gsl_matrix_set(mat_sigma, i, i, std[i]);
  }

  gsl_ran_multivariate_gaussian(pRng, vec_mu, mat_sigma, vec_result);
  for (size_t i = 0; i < N; i++)
    res[i] = gsl_vector_get(vec_result, i);
  double b = 2;
}

inline double Rng::BetaBinomial(const double & mu, const double & nu, const double & n)
{
  if (n < 20)
  {
    return n;
  }
  else if(mu <1e-4)
  {
    return mu;
  }
  else
  {
    double gamma = nu/(n-1);
    double a = (1/gamma - 1)*mu;
    double b = (gamma-1)*(mu-1)/gamma;
    double p;
    if (b/a > 10e2)
    {
      return 0;
    }
    else if (a/b > 10e2)
    {
        return n;
    }
    else 
    {
      double p = gsl_ran_beta(pRng, a, b);
      return gsl_ran_binomial(pRng, p, n);
    }
  }
}

inline void Rng::Multinomial(long n, long k, const double* w, unsigned * X)
{
  gsl_ran_multinomial(pRng, n, k, w, X);
}

inline double Rng::Uniform(const double &a, const double &b)
{
  return (b-a)*gsl_rng_uniform(pRng);
}

#endif