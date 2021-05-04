#include "SIR_Stochastic.hh"

using namespace std;

gsl_vector *SIR_Model::vec_prior_mu;
gsl_vector *SIR_Model::vec_oldProp;
gsl_vector *SIR_Model::vec_resProp;
gsl_matrix *SIR_Model::mat_prior_std;
gsl_matrix *SIR_Model::mat_prop_std;
gsl_vector *SIR_Model::vec_prop_mu;
gsl_vector *SIR_Model::vec_prior_work;

double *SIR_Model::y;
pSIR SIR_Model::x0;
long SIR_Model::N_iterations;
double SIR_Model::ll_std;
double SIR_Model::dt;
double SIR_Model::N_pop;
double SIR_Model::nu_I;
double SIR_Model::nu_R;
bool SIR_Model::is_dispersed;

void beta_binomial(const double & mu, const double & nu, const double & n, double & res, smc::rng* pRng)
{
  if (n < 20)
  {
    res = n;
  }
  else if(mu <1e-4)
  {
    res = mu;
  }
  else
  {
    double gamma = nu/(n-1);
    double a = (1/gamma - 1)*mu;
    double b = (gamma-1)*(mu-1)/gamma;
    double p;
    if (b/a > 10e2)
    {
      res = 0;
    }
    else if (a/b > 10e2)
    {
      res =  n;
    }
    else 
    {
      // std::cout << a << "\t" << b << std::endl;
      double p = gsl_ran_beta(pRng->GetRaw(), a, b);
      res = gsl_ran_binomial(pRng->GetRaw(), p, n);
    }
  }
}

SIR_Model::SIR_Model(long N_iterations, double *y_obs, double *x_init, double DT,double Npop, double* prop_std, double llstd)
{
  vec_prior_mu = gsl_vector_alloc(3);
  vec_oldProp = gsl_vector_alloc(3);
  vec_resProp = gsl_vector_alloc(3);
  vec_prop_mu = gsl_vector_alloc(3);
  vec_prior_work = gsl_vector_alloc(3);
  y = new double[N_iterations];
  memcpy(y, y_obs, sizeof(double) * N_iterations);
  memcpy(x0.X, x_init, sizeof(double)*3);
  mat_prior_std = gsl_matrix_alloc(3, 3);
  mat_prop_std = gsl_matrix_alloc(3, 3);
  ll_std = llstd;
  for (int i = 0; i < 3; i++)
  {
    gsl_matrix_set(mat_prop_std, i, i, prop_std[i]);
  }
  dt = DT;
  N_pop = Npop;
}

void SIR_Model::set_dispersion_parameters(const double &nu_I_init, const double &nu_R_init){
  nu_I = nu_I_init;
  nu_R = nu_R_init;
  is_dispersed = 1;
}

void SIR_Model::dispersion_set(long disperse_flag)
{
  is_dispersed = disperse_flag;
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
  for (int i = 0; i < 3; i++)
  {
    gsl_vector_set(vec_oldProp, i, oldParam[i]);
  }

  gsl_ran_multivariate_gaussian(pRng->GetRaw(), vec_oldProp, mat_prop_std, vec_resProp);
  for (int i = 0; i < 3; i++)
  {
    res[i] = gsl_vector_get(vec_resProp, i);
  }
}

///A function to initialise double type markov chain-valued particles
/// \param pRng A pointer to the random number generator which is to be used
smc::particle<pSIR>* SIR_Model::init(smc::rng *pRng)
{
  smc::particle<pSIR> *InitParticle = new smc::particle<pSIR>;
  InitParticle->Set(x0, 0);
  return InitParticle;
}




//Calculates the next state and likelihood for that state
void SIR_Model::step(long lTime, smc::particle<pSIR>* pState, double *param, smc::rng *pRng)
{

  double alpha = param[0];
  double beta = param[1];

  pSIR* p = pState->GetValuePointer();

  double p_I = 1 - exp(-beta*p->X[1] / N_pop * dt);
  double p_R = 1 - exp(-alpha * dt);

  double K_SI;
  double K_IR;

  if (is_dispersed)
  {
    beta_binomial(p_I, nu_I, p->X[0], K_SI, pRng);
    beta_binomial(p_R, nu_R, p->X[1], K_IR, pRng);
  }
  else{

    K_SI = gsl_ran_poisson(pRng->GetRaw(), p->X[0] * p_I);
    K_IR = gsl_ran_binomial(pRng->GetRaw(), p_R, (int)p->X[1]);    
  }
  double delta_x[3] = {-K_SI, K_SI - K_IR, K_IR};

  for (int i = 0; i < 3; i++)
  {
    p->X[i] += delta_x[i];
  }
  double p_I_y = 1 - exp(-beta*y[lTime] / N_pop * dt);

  double ll = log(gsl_ran_gaussian_pdf(y[lTime]-p->X[1], ll_std));
  pState->AddToLogWeight(ll);
}

void SIR_Model::reset(smc::particle<pSIR>* pState)
{
  memcpy(pState->GetValuePointer()->X, &x0, sizeof(double)*3);
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
  gsl_matrix_free(mat_prior_std);
  gsl_matrix_free(mat_prop_std);
}