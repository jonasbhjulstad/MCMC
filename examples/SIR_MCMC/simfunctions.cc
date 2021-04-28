#include <iostream>
#include <cmath>
#include <gsl/gsl_randist.h>

#include "smctc.hh"
#include "simfunctions.hh"

using namespace std;

extern double N_pop, alpha, beta, dt, I0, std_ll, std_alpha, std_beta;
extern long N_iterates, N_MCMC;
extern double y[];
//Allocate space for alpha, beta
gsl_vector* param_prior_mu = gsl_vector_alloc(2);
gsl_matrix* param_prior_std = gsl_matrix_alloc(2,2);


double particle_logLikelihood(long lTime, double I)
{
  double ll = gsl_ran_gaussian_pdf(I-y[lTime%N_iterates], std_ll);
  return ll;
}
void prior_sample(gsl_vector* res, smc::rng* pRng)
{
  gsl_ran_multivariate_gaussian(pRng->GetRaw(), param_prior_mu, param_prior_std, res);
}
double prior_logLikelihood(const gsl_vector* param)
{
  //Workspace for multivariate normal:
  static gsl_vector* work = gsl_vector_alloc(2);
  
  double plog_prior = 0;
  gsl_ran_multivariate_gaussian_log_pdf(param, param_prior_mu,param_prior_std, &plog_prior, work);
  return plog_prior;
}

// Corresponds to q(theta)
void proposal_sample(gsl_vector* &res, const gsl_vector* oldParam, smc::rng* pRng)
{
  gsl_ran_multivariate_gaussian(pRng->GetRaw(), oldParam, param_prior_std, res);
}

double proposal_loglikelihood(const gsl_vector* param_0, const gsl_vector* param_1)
{
  static gsl_vector* work = gsl_vector_alloc(2);
  double res = .0;
  return gsl_ran_multivariate_gaussian_log_pdf(param_0, param_1, param_prior_std, &res, work);
}
double logWeightFactor(long lTime, const pSIR & pState)
{
  
  double I = pState.X[1];
  double I_prev = pState.X_prev[1];


  //Recursive Weight update factors:

  double plog_prior = prior_logLikelihood(pState.param);

  double plog_ll = log(gsl_ran_gaussian_pdf(I-y[lTime%N_iterates], std_ll));


  double p_I_prev = 1-exp(I_prev/N_pop*dt);
  double plog_transition = log(gsl_ran_binomial_pdf(I, p_I_prev, (int) I_prev));

  return plog_transition+plog_ll-plog_prior;
}

///A function to initialise double type markov chain-valued particles
/// \param pRng A pointer to the random number generator which is to be used
smc::particle<pSIR> fInitialise(smc::rng *pRng)
{
  // Create a Markov chain with the appropriate initialisation and then assign that to the particles.
  gsl_vector_set(param_prior_mu, 0, alpha);
  gsl_vector_set(param_prior_mu, 1, beta);

  gsl_matrix_set(param_prior_std, 0, 0, std_alpha);
  gsl_matrix_set(param_prior_std, 0, 1, 0);
  gsl_matrix_set(param_prior_std, 1, 0, 0);
  gsl_matrix_set(param_prior_std, 1, 1, std_beta);
  
  pSIR* initState = new pSIR;

  initState->X[0] = N_pop-I0;
  initState->X[1] = I0;
  initState->X[2] = 0.0;
  initState->X_prev[0] = N_pop-I0;
  initState->X_prev[1] = I0;
  initState->X_prev[2] = 0.0;
  initState->param = gsl_vector_alloc(2);
  gsl_vector_set(initState->param, 0, alpha);
  gsl_vector_set(initState->param, 1, beta);
  initState->ll_log.push_back(1.0);

  return smc::particle<pSIR>(*initState,0);
}

void f_SIR(pSIR* pState, smc::rng *pRng)
{

  double S = pState->X[0];
  double I = pState->X[1];
  double R = pState->X[2];

  double p_I = 1-exp(I/N_pop*dt);
  double p_R = 1-exp(-alpha*dt);

  double K_SI = gsl_ran_poisson(pRng->GetRaw(), S*p_I);
  double K_IR = gsl_ran_binomial(pRng->GetRaw(), p_R, (int) I);

  double delta_x[3] = {- K_SI, K_SI - K_IR, K_IR};

  for (int i=0; i < 3; i++)
  {
    pState->X_prev[i] = pState->X[i];
    pState->X[i] += delta_x[i];
  }
}

void fMove1(long lTime, smc::particle<pSIR> & pFrom, smc::rng *pRng)
{

  //Move
  f_SIR(pFrom.GetValuePointer(), pRng);
  cout << "Time:\t" << lTime << "\tState:\t" << pFrom.GetValue().X[0] << ",\t" << pFrom.GetValue().X[1] << ",\t"<< pFrom.GetValue().X[2];
  cout << "\t R0: \t" << gsl_vector_get(pFrom.GetValue().param, 1)/gsl_vector_get(pFrom.GetValue().param,0) << endl;
  double wk_1 = pFrom.GetLogWeight() + logWeightFactor(lTime, pFrom.GetValue());

  pFrom.SetLogWeight(wk_1);

  return;
}

///An MCMC step suitable for introducing sample diversity
int fMCMC(long lTime, smc::particle<pSIR> & pFrom, smc::rng *pRng)
{
if ((lTime % (N_iterates)) == 0)
{
  pSIR* pState = pFrom.GetValuePointer();
  gsl_vector* propParam = gsl_vector_alloc(2);
  gsl_vector* oldParam = pFrom.GetValue().param;

  double ll_prev = pState->ll_log.back();
  double ll_current = pFrom.GetLogWeight();

  proposal_sample(propParam, oldParam, pRng);

  double ll_prop = proposal_loglikelihood(propParam, oldParam);
  double ll_prop_reverse = proposal_loglikelihood(oldParam, propParam);

  double alpha_Hastings = exp(ll_current - ll_prev + ll_prop_reverse - ll_prop);

  if(alpha_Hastings < 1)
    if (pRng->UniformS() > alpha_Hastings) {
      gsl_vector_free(propParam);
      pState->ll_log.push_back(ll_prev);
      return false;
    }
  gsl_vector_memcpy(oldParam, propParam);
  gsl_vector_free(propParam);
  pState->X[0] = N_pop-I0;
  pState->X[1] = I0;
  pState->X[2] = 0.0;
  pState->X_prev[0] = N_pop-I0;
  pState->X_prev[1] = I0;
  pState->X_prev[2] = 0.0;
  // cout << "Old particle, alpha = " << gsl_vector_get(oldParam, 0) << ", beta = " << gsl_vector_get(oldParam, 1) << endl;

  return true;
}
else
{
  return false;
}
}

