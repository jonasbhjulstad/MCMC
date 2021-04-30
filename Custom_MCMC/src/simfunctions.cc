#include "simfunctions.hh"

using namespace std;

gsl_vector *SIR_Model::vec_prior_mu;
gsl_vector *SIR_Model::vec_oldProp;
gsl_vector *SIR_Model::vec_resProp;
gsl_matrix *SIR_Model::mat_prior_std;
gsl_matrix *SIR_Model::mat_prop_std;
gsl_vector *SIR_Model::vec_prop_mu;
gsl_vector *SIR_Model::vec_prior_work;

long SIR_Model::N_param_ODE;
long SIR_Model::Nx;
double *SIR_Model::y;
double *SIR_Model::x0;
long SIR_Model::N_iterations;

SIR_Model::SIR_Model(long N_ODE_param, long N_x, long N_iterations, double *y_obs, double *x_init, double* prop_std)
{
  N_param_ODE = N_ODE_param;
  vec_prior_mu = gsl_vector_alloc(N_param_ODE);
  vec_oldProp = gsl_vector_alloc(N_param_ODE);
  vec_resProp = gsl_vector_alloc(N_param_ODE);
  vec_prop_mu = gsl_vector_alloc(N_param_ODE);
  vec_prior_work = gsl_vector_alloc(N_param_ODE);
  Nx = N_x;
  y = new double[N_iterations];
  memcpy(y, y_obs, sizeof(double) * N_iterations);
  x0 = new double[Nx];
  memcpy(x0, x_init, sizeof(double) * Nx);
  mat_prior_std = gsl_matrix_alloc(N_param_ODE, N_param_ODE);
  mat_prop_std = gsl_matrix_alloc(N_param_ODE, N_param_ODE);
  for (int i = 0; i < N_param_ODE; i++)
  {
    gsl_matrix_set(mat_prop_std, i, i, prop_std[i]);
  }
}

void SIR_Model::prior_sample(gsl_vector *res, smc::rng *pRng)
{
  gsl_ran_multivariate_gaussian(pRng->GetRaw(), vec_prior_mu, mat_prior_std, res);
}
double SIR_Model::prior_logLikelihood(const gsl_vector *param)
{
  //Workspace for multivariate normal:

  double plog_prior = 0;
  gsl_ran_multivariate_gaussian_log_pdf(param, vec_prior_mu, mat_prior_std, &plog_prior, vec_prior_work);

  return plog_prior;
}

// Corresponds to q(theta)
void SIR_Model::proposal_sample(double *&res, const double *oldParam, smc::rng *pRng)
{
  for (int i = 0; i < N_param_ODE; i++)
  {
    gsl_vector_set(vec_oldProp, i, oldParam[i]);
  }

  gsl_ran_multivariate_gaussian(pRng->GetRaw(), vec_oldProp, mat_prop_std, vec_resProp);
  for (int i = 0; i < N_param_ODE; i++)
  {
    res[i] = gsl_vector_get(vec_resProp, i);
  }
}

///A function to initialise double type markov chain-valued particles
/// \param pRng A pointer to the random number generator which is to be used
smc::particle<pSIR> SIR_Model::init(smc::rng *pRng)
{
  smc::particle<pSIR> *InitParticle = new smc::particle<pSIR>;
  for (int i = 0; i < Nx; i++)
  {
    InitParticle->GetValuePointer()->X[i] = x0[i];
  }
  InitParticle->SetLogWeight(0);
  // memcpy(*InitParticle->GetValuePointer(), x0, Nx);
  return *InitParticle;
}


//Calculates the next state and likelihood for that state
void SIR_Model::step(long lTime, smc::particle<pSIR> &pState, double *param, smc::rng *pRng)
{

  double alpha = param[0];
  double beta = param[1];
  double N_pop = param[2];
  double dt = param[3];

  double *x = (double *)pState.GetValuePointer();

  double p_I = 1 - exp(-beta*x[1] / N_pop * dt);
  double p_R = 1 - exp(-alpha * dt);

  double K_SI = gsl_ran_poisson(pRng->GetRaw(), x[0] * p_I);
  double K_IR = gsl_ran_binomial(pRng->GetRaw(), p_R, (int)x[1]);


  double delta_x[3] = {-K_SI, K_SI - K_IR, K_IR};

  for (int i = 0; i < 3; i++)
  {
    x[i] += delta_x[i];
  }
  // double p_I_y = 1 - exp(-beta*y[lTime] / N_pop * dt);

  // double ll = log(gsl_ran_binomial_pdf(x[1], p_I_y, (int) y[lTime]));
  double ll = log(gsl_ran_gaussian_pdf(x[1]- y[lTime], 5e6));
  pState.AddToLogWeight(ll);
}

void SIR_Model::reset(smc::particle<pSIR> &pState)
{
  memcpy(pState.GetValuePointer()->X, x0, sizeof(double)*Nx);
}

SIR_Model::~SIR_Model()
{
  gsl_vector_free(vec_prior_mu);
  gsl_vector_free(vec_oldProp);
  gsl_vector_free(vec_resProp);
  gsl_vector_free(vec_prop_mu);
  gsl_vector_free(vec_prior_work);
  if (y)
    delete[] y;
  if (x0)
    delete[] x0;
}

