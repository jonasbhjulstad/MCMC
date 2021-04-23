#include <iostream>
#include <cmath>
#include <gsl/gsl_randist.h>

#include "smctc.hh"
#include "SIR_funcs.hh"

using namespace std;

double var_s0 = 4;
double var_u0 = 1;
double var_s  = 0.02;
double var_u  = 0.001;

double scale_y = 0.1;
double nu_y = 10.0;
double Delta = 0.1;
double sigma = 100000;

///The function corresponding to the log likelihood at specified time and position (up to normalisation)

///  \param lTime The current time (i.e. the index of the current distribution)
///  \param X     The state to consider 
double logLikelihood(long lTime, const particle_SIR & pSIR)
{
  return gsl_ran_lognormal_pdf(pSIR.I, pSIR.y[lTime], sigma);
}

///A function to initialise particles

/// \param pRng A pointer to the random number generator which is to be used
smc::particle<particle_SIR> fInitialise(smc::rng *pRng)
{
  particle_SIR pSIR;
  double I0 = 1e7;
  double S0 = 1e8-I0;

  pSIR.S = S0;
  pSIR.I = I0;
  pSIR.R = 0;


  return smc::particle<particle_SIR>(pSIR,1);
}

///The proposal function.

///\param lTime The sampler iteration.
///\param pFrom The particle to move.
///\param pRng  A random number generator.
void fMove(long lTime, smc::particle<particle_SIR> & pFrom, smc::rng *pRng)
{
  particle_SIR * pSIR = pFrom.GetValuePointer();
  double dt = pSIR->param.dt;
  double alpha = pSIR->param.alpha;
double beta = pSIR->param.beta;
  double N_pop = pSIR->param.N_pop;

  double p_I = 1-exp(beta*pSIR->I/N_pop*dt);
  double p_R = 1-exp(-alpha*dt);

  double K_SI = gsl_ran_poisson(pRng->GetRaw(), pSIR->S*p_I);
  double K_IR = gsl_ran_binomial(pRng->GetRaw(), p_R, (int) pSIR->I);

  pSIR->S += pSIR->S - K_SI;
  pSIR->I += pSIR->I + K_SI - K_IR;
  pSIR->R += pSIR->R + K_IR;

  pFrom.AddToLogWeight(logLikelihood(lTime, *pSIR));
}
