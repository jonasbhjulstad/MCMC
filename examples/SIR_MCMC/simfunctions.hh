#include "smctc.hh"

#include "markovchains/markovchain.h"

extern long lIterates;
extern long lNumber;
extern long lChainLength;
extern double dSchedule;
extern double dThreshold;

typedef double State[3];

double logDensity(long lTime, const mChain<State> & X);

smc::particle<mChain<State> > fInitialise(smc::rng *pRng);
long fSelect(long lTime, const smc::particle<mChain<State> > & p, smc::rng *pRng);
void fMove1(long lTime, smc::particle<mChain<State> > & pFrom, smc::rng *pRng);
void fMove2(long lTime, smc::particle<mChain<State> > & pFrom, smc::rng *pRng);
int fMCMC(long lTime, smc::particle<mChain<State> > & pFrom, smc::rng *pRng);


State pIntegrandPS(long lTime, const smc::particle<mChain<State> >& pPos, void* pVoid);
State pWidthPS(long lTime, void* pVoid);
State pIntegrandFS(const mChain<State>& dPos, void* pVoid);
mChain<State> f_SIR(mChain<State> &Chain, smc::rng *pRng)

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

