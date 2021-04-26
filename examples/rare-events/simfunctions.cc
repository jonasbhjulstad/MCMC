#include <iostream>
#include <cmath>
#include <gsl/gsl_randist.h>

#include "smctc.hh"
#include "simfunctions.hh"

using namespace std;

///The function corresponding to the log posterior density at specified time and position

///   \param lTime The current time (i.e. the index of the current distribution)
///    \param X     The state to consider  **/
extern double N_pop, I0, alpha, beta, dt;
double logDensity(long lTime, const mChain<double> & X)
{
  double lp;

  mElement<double> *x = X.GetElement(0);
  mElement<double> *y = x->pNext;
  //Begin with the density exluding the effect of the potential
  lp = log(gsl_ran_ugaussian_pdf(x->value));

  while(y) {
    lp += log(gsl_ran_ugaussian_pdf(y->value - x->value));
    x = y;
    y = x->pNext;
  }

  //Now include the effect of the multiplicative potential function
  lp -= log(1.0 + exp(-(ALPHA(lTime) * (x->value - THRESHOLD) )));
  return lp;
}

///A function to initialise double type markov chain-valued particles
/// \param pRng A pointer to the random number generator which is to be used
smc::particle<mChain<State> > fInitialise(smc::rng *pRng)
{
  // Create a Markov chain with the appropriate initialisation and then assign that to the particles.
  mChain<State> Mc;

  State x = {N_pop, N_pop-I0, 0};
  for(int i = 0; i < PATHLENGTH; i++) {
    Mc.AppendElement(x);
  }

  return smc::particle<mChain<State> >(Mc,0);
}

///A function to select a move randomly
/// \param lTime  The current evolution time of the system
/// \param p      The current position of the particle which is to be moved
/// \param pRng   A pointer to the random number generator which is to be used
long fSelect(long lTime, const smc::particle<mChain<State> > & p, smc::rng *pRng)
{
    return 0;
}

State F_SIR(State x, smc::rng *pRng)
{

  double S = x[0];
  double I = x[1];
  double R = x[2];

  double p_I = 1-exp(I/N_pop*dt);
  double p_R = 1-exp(-alpha*dt);

  double K_SI = gsl_ran_poisson(pRng->GetRaw(), S*p_I);
  double K_IR = gsl_ran_binomial(pRng->GetRaw(), p_R, (int) I);

  return (State) {S - K_SI, I + K_SI - K_IR, R + K_IR};

}

void fMove1(long lTime, smc::particle<mChain<State> > & pFrom, smc::rng *pRng)
{
  mChain<State>* C = pFrom.GetValuePointer();

  pFrom.AddToLogWeight(logLikelihood(lTime, *pSIR));

  return;
}
///Another move function
void fMove2(long lTime, smc::particle<mChain<double> > & pFrom, smc::rng *pRng)
{
  pFrom.SetLogWeight(pFrom.GetLogWeight() + logDensity(lTime,pFrom.GetValue()) - logDensity(lTime-1,pFrom.GetValue()));
}

///An MCMC step suitable for introducing sample diversity
int fMCMC(long lTime, smc::particle<mChain<double> > & pFrom, smc::rng *pRng)
{
  static smc::particle<mChain<double> > pTo;
  
  mChain<double> * pMC = new mChain<double>;

  for(int i = 0; i < pFrom.GetValue().GetLength(); i++) 
    pMC->AppendElement(pFrom.GetValue().GetElement(i)->value + pRng->Normal(0, 0.5));
  pTo.SetValue(*pMC);
  pTo.SetLogWeight(pFrom.GetLogWeight());

  delete pMC;

  double alpha = exp(logDensity(lTime,pTo.GetValue()) - logDensity(lTime,pFrom.GetValue()));
  if(alpha < 1)
    if (pRng->UniformS() > alpha) {
      return false;
    }

  pFrom = pTo;
  return true;
}

///A function to be integrated in the path sampling step.
double pIntegrandPS(long lTime, const smc::particle<mChain<double> >& pPos, void* pVoid)
{
  double dPos = pPos.GetValue().GetTerminal()->value;
  return (dPos - THRESHOLD) / (1.0 + exp(ALPHA(lTime) * (dPos - THRESHOLD)));
}

///A function which gives the width distribution for the path sampling step.
double pWidthPS(long lTime, void* pVoid)
{
  if(lTime > 1 && lTime < lIterates)
    return ((0.5)*double(ALPHA(lTime+1.0)-ALPHA(lTime-1.0)));
  else 
    return((0.5)*double(ALPHA(lTime+1.0)-ALPHA(lTime)) +(ALPHA(1)-0.0));
}

//The final state weighting function -- how likely is a random path from this distribution to hit the rare set...
double pIntegrandFS(const mChain<double>& dPos, void* pVoid)
{
  if(dPos.GetTerminal()->value > THRESHOLD) {
    return (1.0 + exp(-FTIME*(dPos.GetTerminal()->value-THRESHOLD)));
  }
  else
    return 0;
}

