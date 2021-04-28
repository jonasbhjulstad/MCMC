#include "smctc.hh"
#include <vector>
#include "markovchains/markovchain.h"

extern long lIterates;
extern long lNumber;
extern long lChainLength;
extern double dSchedule;
extern double dThreshold;

struct pSIR
{
    double X[3];
    double X_prev[3];
    gsl_vector* param;
    vector<double> ll_log;
};


smc::particle<pSIR> fInitialise(smc::rng *pRng);
void fMove1(long lTime, smc::particle<pSIR>  & pFrom, smc::rng *pRng);
int fMCMC(long lTime, smc::particle<pSIR> & pFrom, smc::rng *pRng);

double logWeightFactor(long lTime, const pSIR & pState);
double prior_logLikelihood(const gsl_vector* param);
double particle_logLikelihood(long lTime, double I);
void proposal_sample(gsl_vector* &res, const gsl_vector* oldParam, smc::rng* pRng);
// State pIntegrandPS(long lTime, const smc::particle<mChain<pSIR> >& pPos, void* pVoid);
// State pWidthPS(long lTime, void* pVoid);
// State pIntegrandFS(const mChain<pSIR>& dPos, void* pVoid);
void f_SIR(const pSIR* Chain, smc::rng *pRng);
///The number of grid elements to either side of the current state for the single state move
#define GRIDSIZE 12
///The value of alpha at the specified time
#define ALPHA(T) (double(T)*double(dSchedule) / double(lIterates))
///The terminal version of alpha
#define FTIME    (ALPHA(lIterates))
///The exceedance threshold which we are interested in.
#define THRESHOLD dThreshold
///The number of steps in the Markov chain
#define PATHLENGTH lChainLength

