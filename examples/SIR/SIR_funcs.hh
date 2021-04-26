#include "smctc.hh"

typedef struct SIR_params{
double alpha, beta, N_pop, dt;
} SIR_param;

class particle_SIR
{
public:
  double S, I, R;
  double** data;
  SIR_param param;
};

double logLikelihood(long lTime, const double & X);

smc::particle<particle_SIR> fInitialise(smc::rng *pRng);
long fSelect(long lTime, const smc::particle<particle_SIR> & p, 
	     smc::rng *pRng);
void fMove(long lTime, smc::particle<particle_SIR> & pFrom, 
	   smc::rng *pRng);

long load_data(char const * szName, double** ptr_data);

extern double nu_x;
extern double nu_y;
extern double Delta;

extern double * y; 
