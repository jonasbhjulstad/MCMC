#include "simfunctions.hh"

using namespace std;

extern long N_iterates, N_MCMC, Nx, N_sysparam;
extern double prop_mu[], prop_std[];
extern double y[];
extern double x0[];
//Allocate space for alpha, beta

gsl_vector* vec_prop_mu;
gsl_matrix* mat_prop_std;
gsl_vector* vec_prior_mu;
gsl_matrix* mat_prior_std;

// double particle_logLikelihood(long lTime, double I)
// {
//   double ll = gsl_ran_gaussian_pdf(I-y[lTime%N_iterates], std_ll);
//   return ll;
// }
void prior_sample(gsl_vector* res, smc::rng* pRng)
{
  gsl_ran_multivariate_gaussian(pRng->GetRaw(), vec_prior_mu, mat_prior_std, res);
}
double prior_logLikelihood(const gsl_vector* param)
{
  //Workspace for multivariate normal:
  gsl_vector* work = gsl_vector_alloc(2);
  
  double plog_prior = 0;
  gsl_ran_multivariate_gaussian_log_pdf(param, vec_prior_mu,mat_prior_std, &plog_prior, work);

  gsl_vector_free(work);
  return plog_prior;
}

// Corresponds to q(theta)
void proposal_sample(double *&res, const double *oldParam, smc::rng *pRng)
{
  gsl_vector* oldP = gsl_vector_alloc(Nx);
  gsl_vector* result = gsl_vector_alloc(Nx);
  for (int i=0; i < Nx; i++)
  {
    gsl_vector_set(oldP, i, oldParam[i]);
  }

  gsl_ran_multivariate_gaussian(pRng->GetRaw(), oldP, mat_prop_std, result);
  for (int i = 0; i < Nx; i++)
  {
    res[i] = gsl_vector_get(result, i);
  }
  gsl_vector_free(oldP);
  gsl_vector_free(result);

}

// double proposal_loglikelihood(const gsl_vector* param_0, const gsl_vector* param_1)
// {
//   gsl_vector* work = gsl_vector_alloc(2);
//   double res = .0;

//   return gsl_ran_multivariate_gaussian_log_pdf(param_0, param_1, mat_prop_std, &res, work);

// }

///A function to initialise double type markov chain-valued particles
/// \param pRng A pointer to the random number generator which is to be used
smc::particle<pSIR> fInitialise(smc::rng *pRng)
{
  mat_prop_std = gsl_matrix_alloc(N_sysparam, N_sysparam);
  vec_prop_mu = gsl_vector_alloc(N_sysparam);
  for (int i = 0; i < N_sysparam; i++)
  {
    gsl_matrix_set(mat_prop_std, i, i, prop_std[i]);
    gsl_vector_set(vec_prop_mu, i, prop_mu[i]);
  }
  smc::particle<pSIR>* InitParticle = new smc::particle<pSIR>;
  for (int i = 0; i < Nx; i++)
  {
    InitParticle->GetValuePointer()->X[i] = x0[i];
  }
  // memcpy(*InitParticle->GetValuePointer(), x0, Nx);
  return *InitParticle;
}

//Calculates the next state and likelihood for that state
void f_SIR(long lTime, smc::particle<pSIR> &pState, double *param, smc::rng *pRng)
{

  double alpha = param[0];
  double beta = param[1];
  double N_pop = param[2];
  double dt = param[3];


  double* x = (double*) pState.GetValuePointer();

  double p_I = 1-exp(x[1]/N_pop*dt);
  double p_R = 1-exp(-alpha*dt);

  double K_SI = gsl_ran_poisson(pRng->GetRaw(), x[0]*p_I);
  double K_IR = gsl_ran_binomial(pRng->GetRaw(), p_R, (int) x[1]);

  double delta_x[3] = {- K_SI, K_SI - K_IR, K_IR};

  for (int i = 0; i < 3; i++)
  {
    x[i]+= delta_x[i];
  }
  double p_I_y = 1-exp(y[lTime]/N_pop*dt);
  double ll = log(gsl_ran_binomial_pdf(x[1], p_I_y, (int)y[lTime]));
  pState.SetLogWeight(ll);
}
